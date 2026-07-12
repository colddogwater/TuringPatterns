#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <string>
#include <cstring>
#include <algorithm>

#include "libs/cribArray.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "libs/stb_image_write.h"
#include "raylib.h"

#define RAYGUI_IMPLEMENTATION
#include "libs/raygui.h"

using namespace std;

struct UIParameter {
    const char* name;
    float value;
    float minVal;
    float maxVal;

    char text[16];
    bool editMode;

    UIParameter(const char* paramName, float startVal, float min, float max) {
        name = paramName;
        value = startVal;
        minVal = min;
        maxVal = max;
        editMode = false;
        
        snprintf(text, sizeof(text), "%0.3f", value);
    }
};

Color get_heatmap_color(double v) {
    v = std::max(0.0, std::min(1.0, v)); 
    if (v < 0.5) {
        return Color{0, (unsigned char)(v * 2.0 * 255), (unsigned char)((1.0 - v * 2.0) * 255), 255}; // Blue to Green
    } else {
        return Color{(unsigned char)((v - 0.5) * 2.0 * 255), (unsigned char)((1.0 - (v - 0.5) * 2.0) * 255), 0, 255}; // Green to Red
    }
}

void save_heatmap_png(const std::vector<std::vector<float>>& u, const std::string& filename) {
    int height = u.size();
    int width = u[0].size();

    std::vector<Color> pixels(width * height);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            pixels[i * width + j] = get_heatmap_color(u[i][j]);
        }
    }

    stbi_write_png(filename.c_str(), width, height, 4, pixels.data(), width * 4);
    
    //std::cout << "Saved Heatmap to " << filename << "!\n";
}

void initGrid(std::vector<std::vector<float>>& u, std::vector<std::vector<float>>& v, int W, int H) {
    // Initiliase the vector
    int lowH = (H / 2) - 9;
    int highH = (H / 2) + 10;
    int lowW = (W / 2) - 9;
    int highW = (W / 2) + 10;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0, 0.1);

    for (int i = lowH; i < highH; i++) {
        for (int j = lowW; j < highW; j++) {
            u[i][j] = 0.5 + dis(gen);
            v[i][j] = 0.25 + dis(gen);
        }
    }
}

void betterSlider(UIParameter& param, int x, int y) {
    
    if (!param.editMode) {
        snprintf(param.text, sizeof(param.text), "%0.3f", param.value);
    }

    GuiSlider(Rectangle{ (float)x, (float)y, 120, 20 }, param.name, NULL, &param.value, param.minVal, param.maxVal);

    if (GuiTextBox(Rectangle{ (float)(x + 128), (float)y, 60, 20 }, param.text, 16, param.editMode)) {
        param.editMode = !param.editMode;
        
        if (!param.editMode) {
            try {
                param.value = std::stof(param.text);
            } catch (...) {
                param.value = param.minVal; 
            }
            
            // Clamp the math
            if (param.value < param.minVal) param.value = param.minVal;
            if (param.value > param.maxVal) param.value = param.maxVal;
        }
    }
}

int main()
{
    int W = 100;
    int H = 100;
    int t_steps = 12000;
    float dt = 1.0;
    float diffu;
    float diffv;
    static bool showUI = false;

    char textF[16] = "0.035";
    bool editModeF = false;

    // ===== PARAMETERS =====
    float Du = 0.16; // Diffusion Rate of Activator u
    float Dv = 0.08; // Diffusion Rate of Inhibitor v
    float f = 0.035; // Feed Rate
    float k = 0.0625; // Kill Rate
    std::vector<UIParameter> params = {
        UIParameter("f", 0.035f, 0.01f, 0.06f),
        UIParameter("k", 0.0625f, 0.04f, 0.07f),
        UIParameter("Du",   0.16f,   0.01f,  0.3f),
        UIParameter("Dv",   0.08f,   0.01f,  0.16f)
    };
    
    std::vector<std::vector<float>> u(H, std::vector<float>(W, 1));
    std::vector<std::vector<float>> v(H, std::vector<float>(W));

    initGrid(u, v, W, H);

    // Create temp arrays to update
    std::vector<std::vector<float>> Tu(H, std::vector<float>(W, 1));
    std::vector<std::vector<float>> Tv(H, std::vector<float>(W));

    //printArray(u, H, W);
    //save_heatmap_png(u, "numero.png");

    // Window setup
    int scale = 6; 
    InitWindow(W * scale, H * scale, "Turing Pattern Simulation");
    SetTargetFPS(60);

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // The main update loop
    while (!WindowShouldClose()) {
        for (int XD = 0; XD < 5; XD++){
            for (int i = 0; i < H; i++) {
                for (int j = 0; j < W; j++) {
                    f = params[0].value;
                    k = params[1].value;
                    Du = params[2].value;
                    Dv = params[3].value;

                    // --- Diffusion 
                    // determine the neighbouring indices, von neumann neighbours
                    // using periodic boundary conditions too
                    
                    // top + bottom + right + left - 4 middle
                    diffu = Du * (u[(i+1)%H][j] + u[(i-1+H)%H][j] + u[i][(j+1)%W] + u[i][(j-1+W)%W] - 4*u[i][j]);
                    diffv = Dv * (v[(i+1)%H][j] + v[(i-1+H)%H][j] + v[i][(j+1)%W] + v[i][(j-1+W)%W] - 4*v[i][j]);
                    
                    Tu[i][j] = u[i][j] + (diffu + f * (1 - u[i][j]) - u[i][j] * v[i][j] * v[i][j]) * dt;
                    Tv[i][j] = v[i][j] + (diffv + u[i][j] * v[i][j] * v[i][j] - (f + k) * v[i][j]) * dt;
                }
            }
        
        std::swap(u, Tu);
        std::swap(v, Tv);
        }
        
        // ========== FRAME DRAWING
        BeginDrawing();
        ClearBackground(BLACK);

        // --- draw heatmap
        for (int i = 0; i < H; ++i) {
            for (int j = 0; j < W; ++j) {
                if (u[i][j] > 0.01) { 
                    Color cellColor = get_heatmap_color(u[i][j]);
                    
                    // draw to pixel
                    DrawRectangle(j * scale, i * scale, scale, scale, cellColor);
                }
            }
        }
       
        // ====== UI DRAWING
        // collapsable menu button
        if (GuiButton(Rectangle{ 10, 10, 30, 30 }, showUI ? "#120#" : "#121#")) {
            showUI = !showUI; 
        }
        //toggleable menu to control parameters
        if(showUI){
            DrawRectangle(10, 45, 250, 180, Fade(BLACK, 0.7f));

            int startY = 60;
            for (int i = 0; i < params.size(); ++i) {
                // Draw each one, moving down 30 pixels each time
                betterSlider(params[i], 60, startY + (i * 30)); 
            }

            // injects some activator into the center
            if (GuiButton(Rectangle{40, 185, 100, 20}, "#156# Inject")) {
                initGrid(u, v, W, H);
            }
            // resets the grid to the original state
            if (GuiButton(Rectangle{160, 185, 100, 20}, "#77# Reset")) {
                // fill u with ones
                for (auto& row : u) {
                    std::fill(row.begin(), row.end(), 1.0);
                }
                // fill v with zeros
                for (auto& row : v) {
                    std::fill(row.begin(), row.end(), 0.0);
                }
                initGrid(u, v, W, H);
            }
        }
        EndDrawing();
        // ==========

        //std::string filename = "scs/hm" + std::to_string(t) + ".png";
        //printArray(u, H, W);
        
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);

    std::cout << "Simulation took: " << duration.count() << " seconds\n";

    save_heatmap_png(u, "final.png");
    
    CloseWindow();
    return 0;
}