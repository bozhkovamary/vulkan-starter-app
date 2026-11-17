// =============================================================
// application.cpp — Лабораторная работа №3: Текстурирование
// Выполнены базовое задание + все дополнительные задания (на 5):
// 1. Вершинные, индексные буферы и текстуры в VRAM (через Staging буфер).
// 2. Один динамический UBO (VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC).
// 3. Несколько объектов, выбор текстур/материалов из GUI, свой DescriptorSet на материал.
// 4. Нетривиальное сэмплирование (параметризовано от времени, управление через Push Constants).
// 5. Поддержка Albedo + Specular + Emissive (шум Перлина) и служебной черной текстуры 1x1.
// =============================================================
#include "application.hpp"
#include <imgui.h>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstddef>
#include <cstdint>
#include <string>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <algorithm>

#include <vk_mem_alloc.h>

#if __has_include(<stb_image.h>)
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define HAS_STB_IMAGE 1
#elif __has_include("stb_image.h")
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define HAS_STB_IMAGE 1
#else
#define HAS_STB_IMAGE 0
#endif

namespace application {

    namespace {
        // ============================================================
        // 1. ГЕОМЕТРИЯ КУБА С НОРМАЛЯМИ И UV-КООРДИНАТАМИ (24 вершины)
        // ============================================================
        struct Vertex {
            float position[3];
            float normal[3];
            float uv[2];
        };

        const std::vector<Vertex> vertices = {
            // Передняя грань (+Z)
            {{-1,-1, 1}, { 0, 0, 1}, {0.0f, 1.0f}},
            {{ 1,-1, 1}, { 0, 0, 1}, {1.0f, 1.0f}},
            {{ 1, 1, 1}, { 0, 0, 1}, {1.0f, 0.0f}},
            {{-1, 1, 1}, { 0, 0, 1}, {0.0f, 0.0f}},
            // Задняя (-Z)
            {{ 1,-1,-1}, { 0, 0,-1}, {0.0f, 1.0f}},
            {{-1,-1,-1}, { 0, 0,-1}, {1.0f, 1.0f}},
            {{-1, 1,-1}, { 0, 0,-1}, {1.0f, 0.0f}},
            {{ 1, 1,-1}, { 0, 0,-1}, {0.0f, 0.0f}},
            // Правая (+X)
            {{ 1,-1, 1}, { 1, 0, 0}, {0.0f, 1.0f}},
            {{ 1,-1,-1}, { 1, 0, 0}, {1.0f, 1.0f}},
            {{ 1, 1,-1}, { 1, 0, 0}, {1.0f, 0.0f}},
            {{ 1, 1, 1}, { 1, 0, 0}, {0.0f, 0.0f}},
            // Левая (-X)
            {{-1,-1,-1}, {-1, 0, 0}, {0.0f, 1.0f}},
            {{-1,-1, 1}, {-1, 0, 0}, {1.0f, 1.0f}},
            {{-1, 1, 1}, {-1, 0, 0}, {1.0f, 0.0f}},
            {{-1, 1,-1}, {-1, 0, 0}, {0.0f, 0.0f}},
            // Верхняя (+Y)
            {{-1, 1, 1}, { 0, 1, 0}, {0.0f, 1.0f}},
            {{ 1, 1, 1}, { 0, 1, 0}, {1.0f, 1.0f}},
            {{ 1, 1,-1}, { 0, 1, 0}, {1.0f, 0.0f}},
            {{-1, 1,-1}, { 0, 1, 0}, {0.0f, 0.0f}},
            // Нижняя (-Y)
            {{-1,-1,-1}, { 0,-1, 0}, {0.0f, 1.0f}},
            {{ 1,-1,-1}, { 0,-1, 0}, {1.0f, 1.0f}},
            {{ 1,-1, 1}, { 0,-1, 0}, {1.0f, 0.0f}},
            {{-1,-1, 1}, { 0,-1, 0}, {0.0f, 0.0f}},
        };

        const std::vector<uint32_t> indices = {
            0,1,2, 2,3,0,       // передняя
            4,5,6, 6,7,4,       // задняя
            8,9,10, 10,11,8,    // правая
            12,13,14, 14,15,12, // левая
            16,17,18, 18,19,16, // верхняя
            20,21,22, 22,23,20, // нижняя
        };

        // ============================================================
        // 2. СТРУКТУРЫ ДАННЫХ ДЛЯ ШЕЙДЕРОВ
        // ============================================================
        struct SceneUniforms {
            float view_projection[16];
            float camera_position[4];
            float directional_light_direction[4];
            float directional_light_color[4];
            float ambient_light[4];
            uint32_t point_light_count;
            uint32_t spot_light_count;
            uint32_t _pad[2];
        };

        // Структура одного элемента в динамическом UBO
        struct ModelUniforms {
            float model[16];        // 64 B: матрица модели
            float albedo[4];        // 16 B: цвет-множитель (или цвет источника для unlit)
            float specular[4];      // 16 B: цвет блика (rgb) + shininess (.a)
            uint32_t flags[4];      // 16 B: flags[0] = 1 (unlit), 0 (освещаемый по Блинн-Фонгу)
        };

        struct PointLight {
            float position[4];
            float color[4];
        };

        struct SpotLight {
            float position[4];
            float direction[4];
            float color[4];
            float angles[4];
        };

        // Push-константы: только для контроля нетривиального сэмплирования и времени
        struct PipelineConsts {
            uint32_t nontrivial_sampling;
            float time;
            float emissive_strength;
            float wave_strength;
        };

        // ============================================================
        // 3. СЦЕНА, ОБЪЕКТЫ И GUI-СОСТОЯНИЕ
        // ============================================================
        struct SceneObject {
            std::string name;
            float position[3];
            float rotation_deg[3];
            float scale[3];
            float tint[3];
            int material_index;
            bool nontrivial_sampling;
            bool animate_rotation;
        };

        std::vector<SceneObject> scene_objects = {
            { "Main Cube",      { -1.4f, 0.0f,  0.0f }, { 0.0f, 15.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, { 1.0f, 1.0f, 1.0f }, 2, false, false },
            { "Nontrivial Cube",{  1.6f, 0.0f,  0.0f }, { 0.0f,  0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, { 1.0f, 1.0f, 1.0f }, 3, true,  true  },
            { "Floor Plane",    {  0.0f,-1.3f,  0.0f }, { 0.0f,  0.0f, 0.0f }, { 5.0f, 0.15f, 5.0f }, { 1.0f, 1.0f, 1.0f }, 0, false, false },
        };

        float camera_position[3] = { 0.0f, 2.2f, 6.8f };
        float camera_yaw = 0.0f;
        float camera_pitch = -0.25f;
        float camera_fov_deg = 60.0f;

        float dir_light_direction[3] = { -0.5f, -1.0f, -0.3f };
        float dir_light_color[3] = { 0.8f, 0.8f, 0.8f };
        float ambient_color[3] = { 0.08f, 0.08f, 0.12f };

        float material_specular[3] = { 1.0f, 1.0f, 1.0f };
        float material_shininess = 64.0f;
        float emissive_strength = 1.5f;
        float wave_strength = 0.04f;

        std::vector<PointLight> point_lights = {
            {{ 2.0f, 2.2f, 2.0f, 0.0f}, {1.0f, 0.6f, 0.2f, 1.0f}},
        };
        std::vector<SpotLight> spot_lights = {
            {{-1.4f, 3.2f, 2.5f, 0.0f}, {0.0f, -1.0f, -0.6f, 0.0f}, {0.4f, 0.7f, 1.0f, 1.0f}, {18.0f, 32.0f, 0.0f, 0.0f}},
        };

        bool show_light_viz = true;
        float current_time_sec = 0.0f;

        // ============================================================
        // 4. VULKAN-ОБЪЕКТЫ И РЕСУРСЫ ТЕКСТУР
        // ============================================================
        struct TextureGPU {
            VkImage image = VK_NULL_HANDLE;
            VmaAllocation alloc = VK_NULL_HANDLE;
            VkImageView view = VK_NULL_HANDLE;
            uint32_t width = 0;
            uint32_t height = 0;
        };

        struct Material {
            std::string name;
            int albedo_tex_idx = 0;
            int specular_tex_idx = 0;
            int emissive_tex_idx = 0;
            VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
        };

        std::vector<TextureGPU> textures_gpu;
        std::vector<Material> materials;
        VkSampler texture_sampler = VK_NULL_HANDLE;

        // Вершинный и индексный буферы (в VRAM!)
        VkBuffer      vertex_buffer = VK_NULL_HANDLE;
        VmaAllocation vertex_buffer_alloc = VK_NULL_HANDLE;
        VkBuffer      index_buffer = VK_NULL_HANDLE;
        VmaAllocation index_buffer_alloc = VK_NULL_HANDLE;

        // Scene UBO (set 0, binding 0)
        VkBuffer      scene_ubo = VK_NULL_HANDLE;
        VmaAllocation scene_ubo_alloc = VK_NULL_HANDLE;
        void*         scene_ubo_mapped = nullptr;

        // Dynamic Model UBO (set 1, binding 0)
        constexpr size_t MAX_MODELS = 1024;
        VkBuffer      vk_buffer_model_uniforms = VK_NULL_HANDLE;
        VmaAllocation vk_buffer_model_uniforms_allocation = VK_NULL_HANDLE;
        uint8_t*      vk_buffer_model_uniforms_memory = nullptr;
        VkDeviceSize  vk_buffer_model_uniforms_sizeof = sizeof(ModelUniforms);

        // Storage buffers для источников (set 0, binding 1/2)
        constexpr size_t MAX_POINT_LIGHTS = 32;
        constexpr size_t MAX_SPOT_LIGHTS = 16;

        VkBuffer      point_lights_ssbo = VK_NULL_HANDLE;
        VmaAllocation point_lights_ssbo_alloc = VK_NULL_HANDLE;
        void*         point_lights_ssbo_mapped = nullptr;

        VkBuffer      spot_lights_ssbo = VK_NULL_HANDLE;
        VmaAllocation spot_lights_ssbo_alloc = VK_NULL_HANDLE;
        void*         spot_lights_ssbo_mapped = nullptr;

        // Буфер линий для конусов прожекторов
        struct LineVertex { float position[3]; };
        constexpr size_t MAX_LINE_VERTICES = 4096;
        VkBuffer      line_buffer = VK_NULL_HANDLE;
        VmaAllocation line_buffer_alloc = VK_NULL_HANDLE;
        void*         line_buffer_mapped = nullptr;
        size_t        line_vertices_count = 0;

        // Дескрипторы и пайплайны
        VkDescriptorSetLayout scene_set_layout = VK_NULL_HANDLE;
        VkDescriptorSetLayout model_set_layout = VK_NULL_HANDLE;
        VkDescriptorPool      descriptor_pool = VK_NULL_HANDLE;
        VkDescriptorSet       scene_set = VK_NULL_HANDLE;

        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        VkPipeline       main_pipeline = VK_NULL_HANDLE;
        VkPipeline       line_pipeline = VK_NULL_HANDLE;

        // ============================================================
        // 5. МАТЕМАТИКА И ШУМ ПЕРЛИНА
        // ============================================================
        struct Mat4 { float m[16]; };

        Mat4 identity() {
            Mat4 r{};
            r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
            return r;
        }

        Mat4 multiply(const Mat4& a, const Mat4& b) {
            Mat4 r{};
            for (int c = 0; c < 4; ++c)
                for (int row = 0; row < 4; ++row) {
                    float s = 0.0f;
                    for (int k = 0; k < 4; ++k)
                        s += a.m[k * 4 + row] * b.m[c * 4 + k];
                    r.m[c * 4 + row] = s;
                }
            return r;
        }

        Mat4 translate(float x, float y, float z) {
            Mat4 r = identity(); r.m[12] = x; r.m[13] = y; r.m[14] = z; return r;
        }
        Mat4 scaleM(float x, float y, float z) {
            Mat4 r = identity(); r.m[0] = x; r.m[5] = y; r.m[10] = z; return r;
        }
        Mat4 rotX(float a) { Mat4 r = identity(); float c = std::cos(a), s = std::sin(a); r.m[5] = c; r.m[6] = s; r.m[9] = -s; r.m[10] = c; return r; }
        Mat4 rotY(float a) { Mat4 r = identity(); float c = std::cos(a), s = std::sin(a); r.m[0] = c; r.m[2] = -s; r.m[8] = s; r.m[10] = c; return r; }
        Mat4 rotZ(float a) { Mat4 r = identity(); float c = std::cos(a), s = std::sin(a); r.m[0] = c; r.m[1] = s; r.m[4] = -s; r.m[5] = c; return r; }

        Mat4 buildModelMatrix(const float pos[3], const float rot_deg[3], const float sc[3]) {
            const float DEG2RAD = 3.14159265358979323846f / 180.0f;
            Mat4 T = translate(pos[0], pos[1], pos[2]);
            Mat4 Rx = rotX(rot_deg[0] * DEG2RAD);
            Mat4 Ry = rotY(rot_deg[1] * DEG2RAD);
            Mat4 Rz = rotZ(rot_deg[2] * DEG2RAD);
            Mat4 S = scaleM(sc[0], sc[1], sc[2]);
            return multiply(T, multiply(Rz, multiply(Ry, multiply(Rx, S))));
        }

        Mat4 perspective(float fovy, float aspect, float zn, float zf) {
            Mat4 r{};
            float t = std::tan(fovy * 0.5f);
            r.m[0] = 1.0f / (aspect * t);
            r.m[5] = -1.0f / t;
            r.m[10] = zf / (zn - zf);
            r.m[11] = -1.0f;
            r.m[14] = (zf * zn) / (zn - zf);
            return r;
        }

        Mat4 lookAt(const float eye[3], const float center[3], const float up[3]) {
            float f[3] = { center[0] - eye[0], center[1] - eye[1], center[2] - eye[2] };
            float fl = std::sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]);
            f[0] /= fl; f[1] /= fl; f[2] /= fl;

            float s[3] = { f[1] * up[2] - f[2] * up[1], f[2] * up[0] - f[0] * up[2], f[0] * up[1] - f[1] * up[0] };
            float sl = std::sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
            s[0] /= sl; s[1] /= sl; s[2] /= sl;

            float u[3] = { s[1] * f[2] - s[2] * f[1], s[2] * f[0] - s[0] * f[2], s[0] * f[1] - s[1] * f[0] };

            Mat4 r = identity();
            r.m[0] = s[0]; r.m[1] = u[0]; r.m[2] = -f[0];
            r.m[4] = s[1]; r.m[5] = u[1]; r.m[6] = -f[1];
            r.m[8] = s[2]; r.m[9] = u[2]; r.m[10] = -f[2];
            r.m[12] = -(s[0] * eye[0] + s[1] * eye[1] + s[2] * eye[2]);
            r.m[13] = -(u[0] * eye[0] + u[1] * eye[1] + u[2] * eye[2]);
            r.m[14] = (f[0] * eye[0] + f[1] * eye[1] + f[2] * eye[2]);
            return r;
        }

        Mat4 computeView() {
            float cp = std::cos(camera_pitch);
            float dir[3] = {
                std::sin(camera_yaw) * cp,
                std::sin(camera_pitch),
                -std::cos(camera_yaw) * cp,
            };
            float center[3] = {
                camera_position[0] + dir[0],
                camera_position[1] + dir[1],
                camera_position[2] + dir[2],
            };
            float up[3] = { 0.0f, 1.0f, 0.0f };
            return lookAt(camera_position, center, up);
        }

        // Классический 2D-шум Перлина для генерации Emissive-текстуры
        float fade(float t) { return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f); }
        float lerpF(float a, float b, float t) { return a + t * (b - a); }

        float grad2D(int hash, float x, float y) {
            int h = hash & 7;
            float u = h < 4 ? x : y;
            float v = h < 4 ? y : x;
            return ((h & 1) ? -u : u) + ((h & 2) ? -2.0f * v : 2.0f * v);
        }

        float perlin2D(float x, float y) {
            static const int perm[256] = {
                151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
                190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,88,237,149,56,87,174,
                20,125,136,171,168,68,175,74,165,71,134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,
                230,220,105,92,41,55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,18,
                169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,250,124,123,5,202,38,
                147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,223,183,170,213,119,248,152,
                2,44,154,163,70,221,153,101,155,167,43,172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,
                112,104,218,246,97,228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,
                107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,138,236,205,93,222,
                114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
            };
            auto P = [&](int i) { return perm[i & 255]; };
            int X = (int)std::floor(x) & 255;
            int Y = (int)std::floor(y) & 255;
            x -= std::floor(x);
            y -= std::floor(y);
            float u = fade(x), v = fade(y);
            int A = P(X) + Y, B = P(X + 1) + Y;
            float res = lerpF(
                lerpF(grad2D(P(A), x, y), grad2D(P(B), x - 1.0f, y), u),
                lerpF(grad2D(P(A + 1), x, y - 1.0f), grad2D(P(B + 1), x - 1.0f, y - 1.0f), u),
                v
            );
            return 0.5f * (res + 1.0f);
        }

        float fbmPerlin(float x, float y) {
            float total = 0.0f, amp = 0.5f, freq = 1.0f;
            for (int i = 0; i < 4; ++i) {
                total += perlin2D(x * freq, y * freq) * amp;
                freq *= 2.0f;
                amp *= 0.5f;
            }
            return std::clamp(total, 0.0f, 1.0f);
        }

        // ============================================================
        // 6. РАБОТА С ФАЙЛОВОЙ СИСТЕМОЙ И ЗАГРУЗКА ИЗОБРАЖЕНИЙ
        // ============================================================
        struct RawImage {
            uint32_t width = 0;
            uint32_t height = 0;
            std::vector<uint8_t> pixels; // RGBA8
        };

        // Запись 32-битного BMP на диск (чтобы текстуры физически лежали в файловой системе)
        void writeBMP(const std::string& path, uint32_t w, uint32_t h, const std::vector<uint8_t>& rgba) {
            std::ofstream out(path, std::ios::binary);
            if (!out.is_open()) return;

            uint32_t data_size = w * h * 4;
            uint32_t file_size = 54 + data_size;
            uint8_t header[54]{};
            header[0] = 'B'; header[1] = 'M';
            std::memcpy(&header[2], &file_size, 4);
            uint32_t offset = 54;
            std::memcpy(&header[10], &offset, 4);
            uint32_t dib_size = 40;
            std::memcpy(&header[14], &dib_size, 4);
            int32_t width_i = (int32_t)w;
            int32_t height_i = -(int32_t)h; // top-down BMP
            std::memcpy(&header[18], &width_i, 4);
            std::memcpy(&header[22], &height_i, 4);
            uint16_t planes = 1, bpp = 32;
            std::memcpy(&header[26], &planes, 2);
            std::memcpy(&header[28], &bpp, 2);
            std::memcpy(&header[34], &data_size, 4);

            out.write(reinterpret_cast<const char*>(header), 54);
            std::vector<uint8_t> bgra(data_size);
            for (size_t i = 0; i < (size_t)w * h; ++i) {
                bgra[i * 4 + 0] = rgba[i * 4 + 2]; // B
                bgra[i * 4 + 1] = rgba[i * 4 + 1]; // G
                bgra[i * 4 + 2] = rgba[i * 4 + 0]; // R
                bgra[i * 4 + 3] = rgba[i * 4 + 3]; // A
            }
            out.write(reinterpret_cast<const char*>(bgra.data()), data_size);
        }

        // Чтение изображения из файловой системы (через stb_image или встроенный декодер BMP)
        bool loadImageFromFile(const std::string& path, RawImage& out_img) {
#if HAS_STB_IMAGE
            int w = 0, h = 0, ch = 0;
            stbi_uc* data = stbi_load(path.c_str(), &w, &h, &ch, STBI_rgb_alpha);
            if (data && w > 0 && h > 0) {
                out_img.width = (uint32_t)w;
                out_img.height = (uint32_t)h;
                out_img.pixels.assign(data, data + (size_t)w * h * 4);
                stbi_image_free(data);
                return true;
            }
#endif
            std::ifstream in(path, std::ios::binary);
            if (!in.is_open()) {
                std::cerr << "[app] Не удалось открыть файл текстуры: " << path << "\n";
                return false;
            }
            uint8_t header[54]{};
            in.read(reinterpret_cast<char*>(header), 54);
            if (header[0] != 'B' || header[1] != 'M') return false;

            uint32_t data_offset = 0;
            int32_t w = 0, h = 0;
            uint16_t bpp = 0;
            std::memcpy(&data_offset, &header[10], 4);
            std::memcpy(&w, &header[18], 4);
            std::memcpy(&h, &header[22], 4);
            std::memcpy(&bpp, &header[28], 2);

            bool flip_y = (h > 0);
            uint32_t abs_w = (uint32_t)std::abs(w);
            uint32_t abs_h = (uint32_t)std::abs(h);
            out_img.width = abs_w;
            out_img.height = abs_h;
            out_img.pixels.resize((size_t)abs_w * abs_h * 4, 255);

            in.seekg(data_offset, std::ios::beg);
            uint32_t bytes_per_pixel = bpp / 8;
            uint32_t row_padded = (abs_w * bytes_per_pixel + 3) & (~3u);
            std::vector<uint8_t> row_buf(row_padded);

            for (uint32_t y = 0; y < abs_h; ++y) {
                in.read(reinterpret_cast<char*>(row_buf.data()), row_padded);
                uint32_t dst_y = flip_y ? (abs_h - 1 - y) : y;
                for (uint32_t x = 0; x < abs_w; ++x) {
                    size_t dst_idx = ((size_t)dst_y * abs_w + x) * 4;
                    out_img.pixels[dst_idx + 0] = row_buf[x * bytes_per_pixel + 2]; // R
                    out_img.pixels[dst_idx + 1] = row_buf[x * bytes_per_pixel + 1]; // G
                    out_img.pixels[dst_idx + 2] = row_buf[x * bytes_per_pixel + 0]; // B
                    out_img.pixels[dst_idx + 3] = (bytes_per_pixel == 4) ? row_buf[x * 4 + 3] : 255;
                }
            }
            return true;
        }

        // Создает текстурные файлы в папке ../textures/, если их там еще нет
        void ensureDefaultTexturesOnDisk(const std::string& dir) {
            std::filesystem::create_directories(dir);

            // 0. Служебная текстура с одним черным пикселем (1x1)
            if (!std::filesystem::exists(dir + "/black_1x1.bmp")) {
                writeBMP(dir + "/black_1x1.bmp", 1, 1, { 0, 0, 0, 255 });
            }

            const uint32_t W = 256, H = 256;

            // 1. Кирпичная стена (Albedo)
            if (!std::filesystem::exists(dir + "/brick_albedo.bmp")) {
                std::vector<uint8_t> p(W * H * 4);
                for (uint32_t y = 0; y < H; ++y) {
                    for (uint32_t x = 0; x < W; ++x) {
                        uint32_t row = y / 32;
                        uint32_t bx = (x + (row % 2) * 32) % 64;
                        uint32_t by = y % 32;
                        bool mortar = (bx < 4 || by < 4);
                        float n = perlin2D(x * 0.08f, y * 0.08f);
                        size_t idx = (y * W + x) * 4;
                        if (mortar) {
                            uint8_t c = (uint8_t)(170 + n * 30.0f);
                            p[idx+0] = c; p[idx+1] = c - 5; p[idx+2] = c - 10; p[idx+3] = 255;
                        } else {
                            p[idx+0] = (uint8_t)std::clamp(155.0f + n * 55.0f + (row * 17 % 25), 0.0f, 255.0f);
                            p[idx+1] = (uint8_t)std::clamp(55.0f + n * 30.0f, 0.0f, 255.0f);
                            p[idx+2] = (uint8_t)std::clamp(42.0f + n * 25.0f, 0.0f, 255.0f);
                            p[idx+3] = 255;
                        }
                    }
                }
                writeBMP(dir + "/brick_albedo.bmp", W, H, p);
            }

            // 2. Ящик с металлической рамкой (Albedo + Specular)
            if (!std::filesystem::exists(dir + "/crate_albedo.bmp") ||
                !std::filesystem::exists(dir + "/crate_specular.bmp")) {
                std::vector<uint8_t> alb(W * H * 4), spec(W * H * 4);
                for (uint32_t y = 0; y < H; ++y) {
                    for (uint32_t x = 0; x < W; ++x) {
                        bool border = (x < 28 || x >= W - 28 || y < 28 || y >= H - 28);
                        bool diag = std::abs((int)x - (int)y) < 18 && !border;
                        bool metal = border || diag;
                        size_t idx = (y * W + x) * 4;
                        if (metal) {
                            float n = perlin2D(x * 0.15f, y * 0.15f);
                            uint8_t m = (uint8_t)(110 + n * 45.0f);
                            alb[idx+0] = m; alb[idx+1] = m + 5; alb[idx+2] = m + 12; alb[idx+3] = 255;
                            uint8_t s = (uint8_t)(210 + n * 45.0f);
                            spec[idx+0] = s; spec[idx+1] = s; spec[idx+2] = s; spec[idx+3] = 255;
                        } else {
                            float grain = std::sin(y * 0.35f + perlin2D(x * 0.05f, y * 0.05f) * 6.0f) * 0.5f + 0.5f;
                            alb[idx+0] = (uint8_t)(140 + grain * 55.0f);
                            alb[idx+1] = (uint8_t)(95 + grain * 40.0f);
                            alb[idx+2] = (uint8_t)(48 + grain * 25.0f);
                            alb[idx+3] = 255;
                            spec[idx+0] = 20; spec[idx+1] = 20; spec[idx+2] = 20; spec[idx+3] = 255;
                        }
                    }
                }
                writeBMP(dir + "/crate_albedo.bmp", W, H, alb);
                writeBMP(dir + "/crate_specular.bmp", W, H, spec);
            }

            // 3. Лавовый камень с Emissive-картой из шума Перлина (Albedo + Specular + Emissive)
            if (!std::filesystem::exists(dir + "/lava_albedo.bmp") ||
                !std::filesystem::exists(dir + "/lava_specular.bmp") ||
                !std::filesystem::exists(dir + "/lava_emissive_perlin.bmp")) {
                std::vector<uint8_t> alb(W * H * 4), spec(W * H * 4), emis(W * H * 4);
                for (uint32_t y = 0; y < H; ++y) {
                    for (uint32_t x = 0; x < W; ++x) {
                        float nx = (float)x / W * 6.0f;
                        float ny = (float)y / H * 6.0f;
                        float n = fbmPerlin(nx, ny);
                        // Узкие светящиеся прожилки лавы там, где шум Перлина проходит около 0.5
                        float ridge = 1.0f - std::clamp(std::fabs(n - 0.5f) * 10.0f, 0.0f, 1.0f);
                        float glow = ridge * ridge;

                        size_t idx = (y * W + x) * 4;
                        uint8_t rock = (uint8_t)(35 + n * 55.0f);
                        alb[idx+0] = (uint8_t)std::clamp(rock + glow * 180.0f, 0.0f, 255.0f);
                        alb[idx+1] = (uint8_t)std::clamp(rock + glow * 70.0f, 0.0f, 255.0f);
                        alb[idx+2] = rock;
                        alb[idx+3] = 255;

                        uint8_t sp = (uint8_t)((1.0f - glow) * 140.0f);
                        spec[idx+0] = sp; spec[idx+1] = sp; spec[idx+2] = sp; spec[idx+3] = 255;

                        // Emissive из шума Перлина: светится ярко-оранжевым в трещинах, 0 на камне
                        emis[idx+0] = (uint8_t)(glow * 255.0f);
                        emis[idx+1] = (uint8_t)(std::pow(glow, 1.6f) * 180.0f);
                        emis[idx+2] = (uint8_t)(std::pow(glow, 3.0f) * 60.0f);
                        emis[idx+3] = 255;
                    }
                }
                writeBMP(dir + "/lava_albedo.bmp", W, H, alb);
                writeBMP(dir + "/lava_specular.bmp", W, H, spec);
                writeBMP(dir + "/lava_emissive_perlin.bmp", W, H, emis);
            }

            // 4. Шахматная плитка (Albedo + Specular)
            if (!std::filesystem::exists(dir + "/checker_albedo.bmp") ||
                !std::filesystem::exists(dir + "/checker_specular.bmp")) {
                std::vector<uint8_t> alb(W * H * 4), spec(W * H * 4);
                for (uint32_t y = 0; y < H; ++y) {
                    for (uint32_t x = 0; x < W; ++x) {
                        bool cell = (((x / 32) + (y / 32)) % 2) == 0;
                        size_t idx = (y * W + x) * 4;
                        if (cell) {
                            alb[idx+0] = 230; alb[idx+1] = 235; alb[idx+2] = 240; alb[idx+3] = 255;
                            spec[idx+0] = 240; spec[idx+1] = 240; spec[idx+2] = 240; spec[idx+3] = 255;
                        } else {
                            alb[idx+0] = 45; alb[idx+1] = 115; alb[idx+2] = 165; alb[idx+3] = 255;
                            spec[idx+0] = 35; spec[idx+1] = 35; spec[idx+2] = 35; spec[idx+3] = 255;
                        }
                    }
                }
                writeBMP(dir + "/checker_albedo.bmp", W, H, alb);
                writeBMP(dir + "/checker_specular.bmp", W, H, spec);
            }
        }

        // ============================================================
        // 7. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ VULKAN
        // ============================================================
        std::vector<char> readFile(const std::string& path) {
            std::ifstream file(path, std::ios::ate | std::ios::binary);
            if (!file.is_open()) { std::cerr << "[app] Не найден: " << path << "\n"; return {}; }
            size_t size = static_cast<size_t>(file.tellg());
            std::vector<char> buf(size);
            file.seekg(0); file.read(buf.data(), size);
            return buf;
        }

        VkShaderModule createShaderModule(const std::vector<char>& code) {
            VkShaderModuleCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            info.codeSize = code.size();
            info.pCode = reinterpret_cast<const uint32_t*>(code.data());
            VkShaderModule m = VK_NULL_HANDLE;
            vkCreateShaderModule(graphics::internal::context.device, &info, nullptr, &m);
            return m;
        }

        bool createMappedBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
            VkBuffer& out_buf, VmaAllocation& out_alloc, void** out_mapped) {
            VkBufferCreateInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bi.size = size;
            bi.usage = usage;
            bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            VmaAllocationCreateInfo ai{};
            ai.usage = VMA_MEMORY_USAGE_AUTO;
            ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                | VMA_ALLOCATION_CREATE_MAPPED_BIT;

            VmaAllocationInfo info{};
            if (vmaCreateBuffer(graphics::internal::context.allocator, &bi, &ai,
                &out_buf, &out_alloc, &info) != VK_SUCCESS) return false;
            *out_mapped = info.pMappedData;
            return true;
        }

        bool createVRAMBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
            VkBuffer& out_buf, VmaAllocation& out_alloc) {
            VkBufferCreateInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bi.size = size;
            bi.usage = usage;
            bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            VmaAllocationCreateInfo ai{};
            ai.usage = VMA_MEMORY_USAGE_AUTO; // Память выделяется на GPU (VRAM)

            return vmaCreateBuffer(graphics::internal::context.allocator, &bi, &ai,
                &out_buf, &out_alloc, nullptr) == VK_SUCCESS;
        }

        VkPipelineShaderStageCreateInfo makeStage(VkShaderStageFlagBits stage, VkShaderModule mod) {
            VkPipelineShaderStageCreateInfo s{};
            s.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            s.stage = stage;
            s.module = mod;
            s.pName = "main";
            return s;
        }

    } // namespace

    // ================================================================
    // INITIALIZE
    // ================================================================
    bool initialize() {
        auto& ctx = graphics::internal::context;
        VkDevice device = ctx.device;

        // ---------- 0. Подготовка и чтение текстур из файловой системы ----------
        const std::string tex_dir = "../textures";
        ensureDefaultTexturesOnDisk(tex_dir);

        const std::vector<std::string> tex_files = {
            tex_dir + "/black_1x1.bmp",            // 0: служебная черная 1x1
            tex_dir + "/brick_albedo.bmp",         // 1: кирпич Albedo
            tex_dir + "/crate_albedo.bmp",         // 2: ящик Albedo
            tex_dir + "/crate_specular.bmp",       // 3: ящик Specular
            tex_dir + "/lava_albedo.bmp",          // 4: лава Albedo
            tex_dir + "/lava_specular.bmp",        // 5: лава Specular
            tex_dir + "/lava_emissive_perlin.bmp", // 6: лава Emissive (шум Перлина)
            tex_dir + "/checker_albedo.bmp",       // 7: плитка Albedo
            tex_dir + "/checker_specular.bmp",     // 8: плитка Specular
        };

        std::vector<RawImage> raw_images(tex_files.size());
        for (size_t i = 0; i < tex_files.size(); ++i) {
            if (!loadImageFromFile(tex_files[i], raw_images[i])) {
                raw_images[i].width = 1;
                raw_images[i].height = 1;
                raw_images[i].pixels = { 0, 0, 0, 255 };
            }
        }

        // ---------- 1. Выделение буферов вершин, индексов и текстур в VRAM ----------
        const VkDeviceSize vb_size = vertices.size() * sizeof(Vertex);
        const VkDeviceSize ib_size = indices.size() * sizeof(uint32_t);

        if (!createVRAMBuffer(vb_size,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            vertex_buffer, vertex_buffer_alloc)) {
            std::cerr << "[app] VRAM vertex buffer failed\n"; return false;
        }

        if (!createVRAMBuffer(ib_size,
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            index_buffer, index_buffer_alloc)) {
            std::cerr << "[app] VRAM index buffer failed\n"; return false;
        }

        textures_gpu.resize(raw_images.size());
        VkDeviceSize total_staging_size = ((vb_size + 15) & ~15ull) + ((ib_size + 15) & ~15ull);

        for (size_t i = 0; i < raw_images.size(); ++i) {
            textures_gpu[i].width = raw_images[i].width;
            textures_gpu[i].height = raw_images[i].height;
            VkDeviceSize img_bytes = (VkDeviceSize)raw_images[i].width * raw_images[i].height * 4;
            total_staging_size += (img_bytes + 15) & ~15ull;

            VkImageCreateInfo img_info{};
            img_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            img_info.imageType = VK_IMAGE_TYPE_2D;
            img_info.format = VK_FORMAT_R8G8B8A8_UNORM;
            img_info.extent = { raw_images[i].width, raw_images[i].height, 1 };
            img_info.mipLevels = 1;
            img_info.arrayLayers = 1;
            img_info.samples = VK_SAMPLE_COUNT_1_BIT;
            img_info.tiling = VK_IMAGE_TILING_OPTIMAL;
            img_info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            img_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            img_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

            VmaAllocationCreateInfo img_alloc_info{};
            img_alloc_info.usage = VMA_MEMORY_USAGE_AUTO;

            if (vmaCreateImage(ctx.allocator, &img_info, &img_alloc_info,
                &textures_gpu[i].image, &textures_gpu[i].alloc, nullptr) != VK_SUCCESS) {
                std::cerr << "[app] Failed to create VkImage #" << i << "\n"; return false;
            }

            VkImageViewCreateInfo view_info{};
            view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            view_info.image = textures_gpu[i].image;
            view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
            view_info.format = VK_FORMAT_R8G8B8A8_UNORM;
            view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            view_info.subresourceRange.baseMipLevel = 0;
            view_info.subresourceRange.levelCount = 1;
            view_info.subresourceRange.baseArrayLayer = 0;
            view_info.subresourceRange.layerCount = 1;

            if (vkCreateImageView(device, &view_info, nullptr, &textures_gpu[i].view) != VK_SUCCESS) {
                std::cerr << "[app] Failed to create VkImageView #" << i << "\n"; return false;
            }
        }

        // ---------- 2. Копирование через Staging-буфер в VRAM ----------
        VkBuffer staging_buffer = VK_NULL_HANDLE;
        VmaAllocation staging_alloc = VK_NULL_HANDLE;
        void* staging_mapped = nullptr;

        if (!createMappedBuffer(total_staging_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            staging_buffer, staging_alloc, &staging_mapped)) {
            std::cerr << "[app] Staging buffer failed\n"; return false;
        }

        uint8_t* staging_ptr = static_cast<uint8_t*>(staging_mapped);
        VkDeviceSize staging_offset = 0;

        VkCommandPool transfer_pool = VK_NULL_HANDLE;
        VkCommandPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        pool_info.queueFamilyIndex = ctx.graphics_queue_index;
        vkCreateCommandPool(device, &pool_info, nullptr, &transfer_pool);

        VkCommandBuffer oneshot_cmd = VK_NULL_HANDLE;
        VkCommandBufferAllocateInfo cmd_alloc{};
        cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmd_alloc.commandPool = transfer_pool;
        cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmd_alloc.commandBufferCount = 1;
        vkAllocateCommandBuffers(device, &cmd_alloc, &oneshot_cmd);

        VkCommandBufferBeginInfo begin_oneshot{};
        begin_oneshot.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_oneshot.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(oneshot_cmd, &begin_oneshot);

        // Копируем вершины в staging и записываем команду копирования на GPU
        VkDeviceSize vertices_offset = staging_offset;
        std::memcpy(staging_ptr + vertices_offset, vertices.data(), vb_size);
        staging_offset += (vb_size + 15) & ~15ull;

        VkBufferCopy vb_copy{ vertices_offset, 0, vb_size };
        vkCmdCopyBuffer(oneshot_cmd, staging_buffer, vertex_buffer, 1, &vb_copy);

        // Копируем индексы в staging и записываем команду копирования на GPU
        VkDeviceSize indices_offset = staging_offset;
        std::memcpy(staging_ptr + indices_offset, indices.data(), ib_size);
        staging_offset += (ib_size + 15) & ~15ull;

        VkBufferCopy ib_copy{ indices_offset, 0, ib_size };
        vkCmdCopyBuffer(oneshot_cmd, staging_buffer, index_buffer, 1, &ib_copy);

        // Копируем все текстуры из staging в VkImage (VRAM)
        for (size_t i = 0; i < raw_images.size(); ++i) {
            VkDeviceSize img_bytes = (VkDeviceSize)raw_images[i].width * raw_images[i].height * 4;
            VkDeviceSize tex_offset = staging_offset;
            std::memcpy(staging_ptr + tex_offset, raw_images[i].pixels.data(), img_bytes);
            staging_offset += (img_bytes + 15) & ~15ull;

            VkImageMemoryBarrier to_transfer{};
            to_transfer.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            to_transfer.srcAccessMask = 0;
            to_transfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            to_transfer.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            to_transfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            to_transfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_transfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_transfer.image = textures_gpu[i].image;
            to_transfer.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

            vkCmdPipelineBarrier(oneshot_cmd,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &to_transfer);

            VkBufferImageCopy region{};
            region.bufferOffset = tex_offset;
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.mipLevel = 0;
            region.imageSubresource.baseArrayLayer = 0;
            region.imageSubresource.layerCount = 1;
            region.imageExtent = { raw_images[i].width, raw_images[i].height, 1 };

            vkCmdCopyBufferToImage(oneshot_cmd, staging_buffer, textures_gpu[i].image,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

            VkImageMemoryBarrier to_shader{};
            to_shader.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            to_shader.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            to_shader.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            to_shader.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            to_shader.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            to_shader.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_shader.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_shader.image = textures_gpu[i].image;
            to_shader.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

            vkCmdPipelineBarrier(oneshot_cmd,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &to_shader);
        }

        vkEndCommandBuffer(oneshot_cmd);

        VkSubmitInfo submit_oneshot{};
        submit_oneshot.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_oneshot.commandBufferCount = 1;
        submit_oneshot.pCommandBuffers = &oneshot_cmd;
        vkQueueSubmit(ctx.graphics_queue, 1, &submit_oneshot, VK_NULL_HANDLE);
        vkQueueWaitIdle(ctx.graphics_queue);

        vkDestroyCommandPool(device, transfer_pool, nullptr);
        vmaDestroyBuffer(ctx.allocator, staging_buffer, staging_alloc);

        // ---------- 3. Сэмплер текстур ----------
        VkSamplerCreateInfo sampler_info{};
        sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sampler_info.magFilter = VK_FILTER_LINEAR;
        sampler_info.minFilter = VK_FILTER_LINEAR;
        sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampler_info.maxAnisotropy = 1.0f;
        sampler_info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        if (vkCreateSampler(device, &sampler_info, nullptr, &texture_sampler) != VK_SUCCESS) {
            std::cerr << "[app] Sampler creation failed\n"; return false;
        }

        // ---------- 4. Scene UBO и Динамический Model UBO ----------
        if (!createMappedBuffer(sizeof(SceneUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            scene_ubo, scene_ubo_alloc, &scene_ubo_mapped)) {
            return false;
        }

        // Вычисляем выравненный сдвиг для динамического UBO по требованиям физического устройства
        VkPhysicalDeviceProperties physical_device_props{};
        vkGetPhysicalDeviceProperties(ctx.physical_device, &physical_device_props);
        const VkDeviceSize dynamic_alignment = physical_device_props.limits.minUniformBufferOffsetAlignment;

        vk_buffer_model_uniforms_sizeof =
            (sizeof(ModelUniforms) + dynamic_alignment - 1) & ~(dynamic_alignment - 1);

        void* dyn_mapped = nullptr;
        if (!createMappedBuffer(vk_buffer_model_uniforms_sizeof * MAX_MODELS,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            vk_buffer_model_uniforms, vk_buffer_model_uniforms_allocation, &dyn_mapped)) {
            std::cerr << "[app] Dynamic model UBO failed\n"; return false;
        }
        vk_buffer_model_uniforms_memory = static_cast<uint8_t*>(dyn_mapped);

        // ---------- 5. Storage buffers для источников и буфер линий ----------
        if (!createMappedBuffer(sizeof(PointLight) * MAX_POINT_LIGHTS,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            point_lights_ssbo, point_lights_ssbo_alloc, &point_lights_ssbo_mapped)) return false;

        if (!createMappedBuffer(sizeof(SpotLight) * MAX_SPOT_LIGHTS,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            spot_lights_ssbo, spot_lights_ssbo_alloc, &spot_lights_ssbo_mapped)) return false;

        if (!createMappedBuffer(sizeof(LineVertex) * MAX_LINE_VERTICES,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            line_buffer, line_buffer_alloc, &line_buffer_mapped)) return false;

        // ---------- 6. Материалы (список текстурных наборов) ----------
        // Индексы текстур:
        // 0 = black_1x1 (служебная черная текстура, когда нет specular или emissive)
        materials = {
            { "Brick Wall (Albedo only)",               1, 0, 0, VK_NULL_HANDLE },
            { "Metal Crate (Albedo + Specular)",        2, 3, 0, VK_NULL_HANDLE },
            { "Lava Stone (Albedo + Spec + Perlin Emissive)", 4, 5, 6, VK_NULL_HANDLE },
            { "Checker Tiles (Albedo + Specular)",      7, 8, 0, VK_NULL_HANDLE },
        };

        // ---------- 7. DESCRIPTOR SET LAYOUTS ----------
        // Set 0: Сцена
        VkDescriptorSetLayoutBinding scene_bindings[3]{};
        scene_bindings[0] = { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
        scene_bindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
        scene_bindings[2] = { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };

        VkDescriptorSetLayoutCreateInfo scene_dsl{};
        scene_dsl.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        scene_dsl.bindingCount = 3;
        scene_dsl.pBindings = scene_bindings;
        if (vkCreateDescriptorSetLayout(device, &scene_dsl, nullptr, &scene_set_layout) != VK_SUCCESS) return false;

        // Set 1: Модель (Динамический UBO + Albedo + Specular + Emissive)
        VkDescriptorSetLayoutBinding model_bindings[4]{};
        model_bindings[0] = { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
        model_bindings[1] = { 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
        model_bindings[2] = { 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
        model_bindings[3] = { 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };

        VkDescriptorSetLayoutCreateInfo model_dsl{};
        model_dsl.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        model_dsl.bindingCount = 4;
        model_dsl.pBindings = model_bindings;
        if (vkCreateDescriptorSetLayout(device, &model_dsl, nullptr, &model_set_layout) != VK_SUCCESS) return false;

        // ---------- 8. DESCRIPTOR POOL & SETS ----------
        VkDescriptorPoolSize pool_sizes[4]{};
        pool_sizes[0] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4 };
        pool_sizes[1] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 };
        pool_sizes[2] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 16 };
        pool_sizes[3] = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 48 };

        VkDescriptorPoolCreateInfo dp{};
        dp.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dp.maxSets = 16;
        dp.poolSizeCount = 4;
        dp.pPoolSizes = pool_sizes;
        if (vkCreateDescriptorPool(device, &dp, nullptr, &descriptor_pool) != VK_SUCCESS) return false;

        // Выделяем и обновляем scene_set
        VkDescriptorSetAllocateInfo ds_alloc{};
        ds_alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ds_alloc.descriptorPool = descriptor_pool;
        ds_alloc.descriptorSetCount = 1;
        ds_alloc.pSetLayouts = &scene_set_layout;
        vkAllocateDescriptorSets(device, &ds_alloc, &scene_set);

        VkDescriptorBufferInfo scene_info{ scene_ubo, 0, sizeof(SceneUniforms) };
        VkDescriptorBufferInfo point_info{ point_lights_ssbo, 0, sizeof(PointLight) * MAX_POINT_LIGHTS };
        VkDescriptorBufferInfo spot_info{ spot_lights_ssbo, 0, sizeof(SpotLight) * MAX_SPOT_LIGHTS };

        VkWriteDescriptorSet scene_writes[3]{};
        scene_writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, scene_set, 0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &scene_info, nullptr };
        scene_writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, scene_set, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &point_info, nullptr };
        scene_writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, scene_set, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &spot_info, nullptr };
        vkUpdateDescriptorSets(device, 3, scene_writes, 0, nullptr);

        // Выделяем и обновляем отдельный набор дескрипторов для каждого материала
        VkDescriptorBufferInfo dyn_ubo_info{};
        dyn_ubo_info.buffer = vk_buffer_model_uniforms;
        dyn_ubo_info.offset = 0;
        dyn_ubo_info.range = sizeof(ModelUniforms);

        for (auto& mat : materials) {
            ds_alloc.pSetLayouts = &model_set_layout;
            vkAllocateDescriptorSets(device, &ds_alloc, &mat.descriptor_set);

            VkDescriptorImageInfo img_albedo{ texture_sampler, textures_gpu[mat.albedo_tex_idx].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
            VkDescriptorImageInfo img_spec{ texture_sampler, textures_gpu[mat.specular_tex_idx].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
            VkDescriptorImageInfo img_emis{ texture_sampler, textures_gpu[mat.emissive_tex_idx].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };

            VkWriteDescriptorSet mwrites[4]{};
            mwrites[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, mat.descriptor_set, 0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, nullptr, &dyn_ubo_info, nullptr };
            mwrites[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, mat.descriptor_set, 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &img_albedo, nullptr, nullptr };
            mwrites[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, mat.descriptor_set, 2, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &img_spec, nullptr, nullptr };
            mwrites[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, mat.descriptor_set, 3, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &img_emis, nullptr, nullptr };
            vkUpdateDescriptorSets(device, 4, mwrites, 0, nullptr);
        }

        // ---------- 9. PIPELINE LAYOUT ----------
        VkDescriptorSetLayout all_sets[] = { scene_set_layout, model_set_layout };

        VkPushConstantRange pc_range{};
        pc_range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        pc_range.offset = 0;
        pc_range.size = (sizeof(PipelineConsts) + sizeof(uint32_t) - 1) & ~(sizeof(uint32_t) - 1);

        VkPipelineLayoutCreateInfo pl_info{};
        pl_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl_info.setLayoutCount = 2;
        pl_info.pSetLayouts = all_sets;
        pl_info.pushConstantRangeCount = 1;
        pl_info.pPushConstantRanges = &pc_range;
        if (vkCreatePipelineLayout(device, &pl_info, nullptr, &pipeline_layout) != VK_SUCCESS) return false;

        // ---------- 10. SHADER MODULES & PIPELINES ----------
        auto vert_code = readFile("../shaders/shader.vert.spv");
        auto frag_code = readFile("../shaders/shader.frag.spv");
        if (vert_code.empty() || frag_code.empty()) return false;

        VkShaderModule vert_mod = createShaderModule(vert_code);
        VkShaderModule frag_mod = createShaderModule(frag_code);
        if (!vert_mod || !frag_mod) return false;

        VkPipelineShaderStageCreateInfo stages[2] = {
            makeStage(VK_SHADER_STAGE_VERTEX_BIT, vert_mod),
            makeStage(VK_SHADER_STAGE_FRAGMENT_BIT, frag_mod),
        };

        VkVertexInputBindingDescription vib{};
        vib.binding = 0;
        vib.stride = sizeof(Vertex);
        vib.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        VkVertexInputAttributeDescription via[3]{};
        via[0] = { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position) };
        via[1] = { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) };
        via[2] = { 2, 0, VK_FORMAT_R32G32_SFLOAT,    offsetof(Vertex, uv) };

        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vi.vertexBindingDescriptionCount = 1;
        vi.pVertexBindingDescriptions = &vib;
        vi.vertexAttributeDescriptionCount = 3;
        vi.pVertexAttributeDescriptions = via;

        VkPipelineViewportStateCreateInfo vp{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, nullptr, 0, 1, nullptr, 1, nullptr };

        VkPipelineRasterizationStateCreateInfo rs{};
        rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = VK_CULL_MODE_BACK_BIT;
        rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo ds{};
        ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds.depthTestEnable = VK_TRUE;
        ds.depthWriteEnable = VK_TRUE;
        ds.depthCompareOp = VK_COMPARE_OP_LESS;

        VkPipelineColorBlendAttachmentState cba{};
        cba.colorWriteMask = 0xF;
        cba.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo cb{};
        cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cb.attachmentCount = 1;
        cb.pAttachments = &cba;

        VkDynamicState dyn_states[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dyn{};
        dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dyn.dynamicStateCount = 2;
        dyn.pDynamicStates = dyn_states;

        VkPipelineInputAssemblyStateCreateInfo ia{};
        ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkGraphicsPipelineCreateInfo pipe{};
        pipe.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipe.stageCount = 2;
        pipe.pStages = stages;
        pipe.pVertexInputState = &vi;
        pipe.pInputAssemblyState = &ia;
        pipe.pViewportState = &vp;
        pipe.pRasterizationState = &rs;
        pipe.pMultisampleState = &ms;
        pipe.pDepthStencilState = &ds;
        pipe.pColorBlendState = &cb;
        pipe.pDynamicState = &dyn;
        pipe.layout = pipeline_layout;
        pipe.renderPass = ctx.render_pass;
        pipe.subpass = 0;

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipe, nullptr, &main_pipeline) != VK_SUCCESS) return false;

        // Line pipeline для конусов прожекторов
        ia.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
        rs.cullMode = VK_CULL_MODE_NONE;

        VkVertexInputBindingDescription line_vib{ 0, sizeof(LineVertex), VK_VERTEX_INPUT_RATE_VERTEX };
        VkVertexInputAttributeDescription line_via[3] = {
            { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0 },
            { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, 0 },
            { 2, 0, VK_FORMAT_R32G32_SFLOAT,    0 },
        };
        VkPipelineVertexInputStateCreateInfo line_vi{};
        line_vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        line_vi.vertexBindingDescriptionCount = 1;
        line_vi.pVertexBindingDescriptions = &line_vib;
        line_vi.vertexAttributeDescriptionCount = 3;
        line_vi.pVertexAttributeDescriptions = line_via;
        pipe.pVertexInputState = &line_vi;

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipe, nullptr, &line_pipeline) != VK_SUCCESS) return false;

        vkDestroyShaderModule(device, vert_mod, nullptr);
        vkDestroyShaderModule(device, frag_mod, nullptr);

        return true;
    }

    // ================================================================
    // SHUTDOWN
    // ================================================================
    void shutdown() {
        auto& ctx = graphics::internal::context;
        vkDeviceWaitIdle(ctx.device);

        vkDestroyPipeline(ctx.device, main_pipeline, nullptr);
        vkDestroyPipeline(ctx.device, line_pipeline, nullptr);
        vkDestroyPipelineLayout(ctx.device, pipeline_layout, nullptr);
        vkDestroyDescriptorPool(ctx.device, descriptor_pool, nullptr);
        vkDestroyDescriptorSetLayout(ctx.device, scene_set_layout, nullptr);
        vkDestroyDescriptorSetLayout(ctx.device, model_set_layout, nullptr);

        vkDestroySampler(ctx.device, texture_sampler, nullptr);
        for (auto& t : textures_gpu) {
            vkDestroyImageView(ctx.device, t.view, nullptr);
            vmaDestroyImage(ctx.allocator, t.image, t.alloc);
        }
        textures_gpu.clear();

        vmaDestroyBuffer(ctx.allocator, line_buffer, line_buffer_alloc);
        vmaDestroyBuffer(ctx.allocator, spot_lights_ssbo, spot_lights_ssbo_alloc);
        vmaDestroyBuffer(ctx.allocator, point_lights_ssbo, point_lights_ssbo_alloc);
        vmaDestroyBuffer(ctx.allocator, vk_buffer_model_uniforms, vk_buffer_model_uniforms_allocation);
        vmaDestroyBuffer(ctx.allocator, scene_ubo, scene_ubo_alloc);
        vmaDestroyBuffer(ctx.allocator, index_buffer, index_buffer_alloc);
        vmaDestroyBuffer(ctx.allocator, vertex_buffer, vertex_buffer_alloc);
    }

    // ================================================================
    // UPDATE
    // ================================================================
    void update(double time_arg) {
        auto& ctx = graphics::internal::context;
        const float DEG2RAD = 3.14159265358979323846f / 180.0f;
        current_time_sec = static_cast<float>(time_arg);

        // ---------- GUI ----------
        ImGui::Begin("Lab #3: Texture Mapping & Dynamic UBO");

        if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::DragFloat3("Position", camera_position, 0.1f);
            ImGui::SliderAngle("Yaw", &camera_yaw, -180.0f, 180.0f);
            ImGui::SliderAngle("Pitch", &camera_pitch, -89.0f, 89.0f);
            ImGui::SliderFloat("FOV", &camera_fov_deg, 20.0f, 120.0f);
        }

        if (ImGui::CollapsingHeader("Models & Materials (Dynamic UBO)", ImGuiTreeNodeFlags_DefaultOpen)) {
            std::vector<const char*> mat_names;
            for (const auto& m : materials) mat_names.push_back(m.name.c_str());

            for (size_t i = 0; i < scene_objects.size(); ++i) {
                ImGui::PushID((int)i + 5000);
                if (ImGui::TreeNode(scene_objects[i].name.c_str())) {
                    ImGui::Combo("Texture / Material", &scene_objects[i].material_index,
                        mat_names.data(), (int)mat_names.size());
                    ImGui::DragFloat3("Position", scene_objects[i].position, 0.1f);
                    ImGui::DragFloat3("Rotation", scene_objects[i].rotation_deg, 1.0f);
                    ImGui::DragFloat3("Scale", scene_objects[i].scale, 0.05f, 0.05f, 20.0f);
                    ImGui::ColorEdit3("Color Tint", scene_objects[i].tint);
                    ImGui::Checkbox("Non-trivial Sampling (Push Const)", &scene_objects[i].nontrivial_sampling);
                    ImGui::Checkbox("Animate Rotation", &scene_objects[i].animate_rotation);
                    ImGui::TreePop();
                }
                ImGui::PopID();
            }

            ImGui::Separator();
            ImGui::ColorEdit3("Global Specular Color", material_specular);
            ImGui::SliderFloat("Shininess", &material_shininess, 2.0f, 256.0f);
            ImGui::SliderFloat("Emissive Strength", &emissive_strength, 0.0f, 5.0f);
            ImGui::SliderFloat("Wave Distortion", &wave_strength, 0.0f, 0.15f);
        }

        if (ImGui::CollapsingHeader("Directional & Ambient Light")) {
            ImGui::DragFloat3("Direction##dir", dir_light_direction, 0.05f);
            ImGui::ColorEdit3("Color##dir", dir_light_color);
            ImGui::ColorEdit3("Ambient color", ambient_color);
        }

        if (ImGui::CollapsingHeader("Point Lights")) {
            if (ImGui::Button("Add Point Light") && point_lights.size() < MAX_POINT_LIGHTS) {
                point_lights.push_back({ {0.0f, 2.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f, 1.0f} });
            }
            for (size_t i = 0; i < point_lights.size(); ++i) {
                ImGui::PushID((int)i);
                if (ImGui::TreeNode("Point Light", "Light #%d", (int)i)) {
                    ImGui::DragFloat3("Position", &point_lights[i].position[0], 0.1f);
                    ImGui::ColorEdit3("Color", &point_lights[i].color[0]);
                    if (ImGui::Button("Remove")) {
                        point_lights.erase(point_lights.begin() + i);
                        ImGui::TreePop(); ImGui::PopID(); break;
                    }
                    ImGui::TreePop();
                }
                ImGui::PopID();
            }
        }

        if (ImGui::CollapsingHeader("Spot Lights")) {
            if (ImGui::Button("Add Spot Light") && spot_lights.size() < MAX_SPOT_LIGHTS) {
                spot_lights.push_back({ {0, 3, 0, 0}, {0, -1, 0, 0}, {1, 1, 1, 1}, {20, 35, 0, 0} });
            }
            for (size_t i = 0; i < spot_lights.size(); ++i) {
                ImGui::PushID((int)i + 1000);
                if (ImGui::TreeNode("Spot Light", "Spot #%d", (int)i)) {
                    ImGui::DragFloat3("Position", &spot_lights[i].position[0], 0.1f);
                    ImGui::DragFloat3("Direction", &spot_lights[i].direction[0], 0.05f);
                    ImGui::ColorEdit3("Color", &spot_lights[i].color[0]);
                    ImGui::SliderFloat("Inner angle", &spot_lights[i].angles[0], 1.0f, 89.0f);
                    ImGui::SliderFloat("Outer angle", &spot_lights[i].angles[1], 1.0f, 89.0f);
                    if (ImGui::Button("Remove")) {
                        spot_lights.erase(spot_lights.begin() + i);
                        ImGui::TreePop(); ImGui::PopID(); break;
                    }
                    ImGui::TreePop();
                }
                ImGui::PopID();
            }
        }

        ImGui::Checkbox("Show light visualization", &show_light_viz);
        ImGui::End();

        // ---------- 1. Обновление Scene UBO ----------
        Mat4 view = computeView();
        float aspect = (float)ctx.swapchain_extent.width / (float)ctx.swapchain_extent.height;
        Mat4 proj = perspective(camera_fov_deg * DEG2RAD, aspect, 0.1f, 100.0f);
        Mat4 view_proj = multiply(proj, view);

        SceneUniforms su{};
        std::memcpy(su.view_projection, view_proj.m, sizeof(view_proj.m));
        su.camera_position[0] = camera_position[0];
        su.camera_position[1] = camera_position[1];
        su.camera_position[2] = camera_position[2];
        su.camera_position[3] = 1.0f;

        su.directional_light_direction[0] = dir_light_direction[0];
        su.directional_light_direction[1] = dir_light_direction[1];
        su.directional_light_direction[2] = dir_light_direction[2];

        std::memcpy(su.directional_light_color, dir_light_color, sizeof(dir_light_color));
        std::memcpy(su.ambient_light, ambient_color, sizeof(ambient_color));
        su.point_light_count = (uint32_t)point_lights.size();
        su.spot_light_count = (uint32_t)spot_lights.size();
        std::memcpy(scene_ubo_mapped, &su, sizeof(su));

        // ---------- 2. Обновление Динамического UBO для всех моделей ----------
        uint32_t model_slot = 0;

        // 2а. Основные объекты сцены
        for (size_t i = 0; i < scene_objects.size() && model_slot < MAX_MODELS; ++i, ++model_slot) {
            float rot[3] = {
                scene_objects[i].rotation_deg[0],
                scene_objects[i].rotation_deg[1] + (scene_objects[i].animate_rotation ? current_time_sec * 35.0f : 0.0f),
                scene_objects[i].rotation_deg[2]
            };
            Mat4 m = buildModelMatrix(scene_objects[i].position, rot, scene_objects[i].scale);

            ModelUniforms* u = reinterpret_cast<ModelUniforms*>(
                vk_buffer_model_uniforms_memory + model_slot * vk_buffer_model_uniforms_sizeof);
            std::memcpy(u->model, m.m, sizeof(m.m));
            u->albedo[0] = scene_objects[i].tint[0];
            u->albedo[1] = scene_objects[i].tint[1];
            u->albedo[2] = scene_objects[i].tint[2];
            u->albedo[3] = 1.0f;
            std::memcpy(u->specular, material_specular, sizeof(material_specular));
            u->specular[3] = material_shininess;
            u->flags[0] = 0u; // Освещается по Блинн-Фонгу
        }

        // 2б. Кубики точечных источников (тоже через динамический UBO!)
        for (const auto& pl : point_lights) {
            if (model_slot >= MAX_MODELS) break;
            Mat4 m = multiply(translate(pl.position[0], pl.position[1], pl.position[2]), scaleM(0.15f, 0.15f, 0.15f));
            ModelUniforms* u = reinterpret_cast<ModelUniforms*>(
                vk_buffer_model_uniforms_memory + model_slot * vk_buffer_model_uniforms_sizeof);
            std::memcpy(u->model, m.m, sizeof(m.m));
            std::memcpy(u->albedo, pl.color, sizeof(pl.color));
            u->flags[0] = 1u; // unlit
            ++model_slot;
        }

        // 2в. Кубики прожекторных источников (через динамический UBO!)
        for (const auto& sl : spot_lights) {
            if (model_slot >= MAX_MODELS) break;
            Mat4 m = multiply(translate(sl.position[0], sl.position[1], sl.position[2]), scaleM(0.15f, 0.15f, 0.15f));
            ModelUniforms* u = reinterpret_cast<ModelUniforms*>(
                vk_buffer_model_uniforms_memory + model_slot * vk_buffer_model_uniforms_sizeof);
            std::memcpy(u->model, m.m, sizeof(m.m));
            std::memcpy(u->albedo, sl.color, sizeof(sl.color));
            u->flags[0] = 1u; // unlit
            ++model_slot;
        }

        // 2г. Слот для белых линий конусов прожекторов (единичная матрица, белый цвет)
        if (model_slot < MAX_MODELS) {
            Mat4 ident = identity();
            ModelUniforms* u = reinterpret_cast<ModelUniforms*>(
                vk_buffer_model_uniforms_memory + model_slot * vk_buffer_model_uniforms_sizeof);
            std::memcpy(u->model, ident.m, sizeof(ident.m));
            u->albedo[0] = 1.0f; u->albedo[1] = 1.0f; u->albedo[2] = 1.0f; u->albedo[3] = 1.0f;
            u->flags[0] = 1u; // unlit
            ++model_slot;
        }

        // ---------- 3. Storage-буферы и генерация линий конусов ----------
        if (!point_lights.empty())
            std::memcpy(point_lights_ssbo_mapped, point_lights.data(), point_lights.size() * sizeof(PointLight));
        if (!spot_lights.empty())
            std::memcpy(spot_lights_ssbo_mapped, spot_lights.data(), spot_lights.size() * sizeof(SpotLight));

        line_vertices_count = 0;
        if (show_light_viz) {
            LineVertex* verts = static_cast<LineVertex*>(line_buffer_mapped);
            constexpr int SEGMENTS = 24;
            constexpr float CONE_LENGTH = 2.0f;
            constexpr float PI = 3.14159265358979323846f;

            for (const auto& sl : spot_lights) {
                if (line_vertices_count + SEGMENTS * 4 > MAX_LINE_VERTICES) break;
                float ox = sl.position[0], oy = sl.position[1], oz = sl.position[2];
                float dx = sl.direction[0], dy = sl.direction[1], dz = sl.direction[2];
                float dl = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (dl < 1e-5f) continue;
                dx /= dl; dy /= dl; dz /= dl;

                float ux, uy, uz;
                if (std::fabs(dx) < 0.9f) { ux = 0.0f; uy = dz; uz = -dy; }
                else { ux = -dz; uy = 0.0f; uz = dx; }
                float ul = std::sqrt(ux * ux + uy * uy + uz * uz);
                ux /= ul; uy /= ul; uz /= ul;

                float vx = dy * uz - dz * uy;
                float vy = dz * ux - dx * uz;
                float vz = dx * uy - dy * ux;

                float radius = CONE_LENGTH * std::tan(sl.angles[1] * PI / 180.0f);
                float bx = ox + dx * CONE_LENGTH, by = oy + dy * CONE_LENGTH, bz = oz + dz * CONE_LENGTH;

                float circle_pts[24 * 3];
                for (int i = 0; i < SEGMENTS; ++i) {
                    float t = (float)i / (float)SEGMENTS * 2.0f * PI;
                    float c = std::cos(t), s = std::sin(t);
                    circle_pts[i * 3 + 0] = bx + radius * (c * ux + s * vx);
                    circle_pts[i * 3 + 1] = by + radius * (c * uy + s * vy);
                    circle_pts[i * 3 + 2] = bz + radius * (c * uz + s * vz);
                }

                for (int i = 0; i < SEGMENTS; ++i) {
                    verts[line_vertices_count++] = { { ox, oy, oz } };
                    verts[line_vertices_count++] = { { circle_pts[i * 3 + 0], circle_pts[i * 3 + 1], circle_pts[i * 3 + 2] } };
                }
                for (int i = 0; i < SEGMENTS; ++i) {
                    int j = (i + 1) % SEGMENTS;
                    verts[line_vertices_count++] = { { circle_pts[i * 3 + 0], circle_pts[i * 3 + 1], circle_pts[i * 3 + 2] } };
                    verts[line_vertices_count++] = { { circle_pts[j * 3 + 0], circle_pts[j * 3 + 1], circle_pts[j * 3 + 2] } };
                }
            }
        }
    }

    // ================================================================
    // RENDER
    // ================================================================
    void render(const graphics::internal::FrameData& fd) {
        auto& ctx = graphics::internal::context;

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(fd.command_buffer, &begin_info);

        VkClearValue clear_values[2] = {};
        clear_values[0].color = { {0.04f, 0.04f, 0.07f, 1.0f} };
        clear_values[1].depthStencil = { 1.0f, 0 };

        VkRenderPassBeginInfo rp{};
        rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rp.renderPass = ctx.render_pass;
        rp.framebuffer = fd.framebuffer;
        rp.renderArea.offset = { 0, 0 };
        rp.renderArea.extent = ctx.swapchain_extent;
        rp.clearValueCount = 2;
        rp.pClearValues = clear_values;

        vkCmdBeginRenderPass(fd.command_buffer, &rp, VK_SUBPASS_CONTENTS_INLINE);

        VkViewport vp{ 0.0f, 0.0f, (float)ctx.swapchain_extent.width, (float)ctx.swapchain_extent.height, 0.0f, 1.0f };
        vkCmdSetViewport(fd.command_buffer, 0, 1, &vp);

        VkRect2D sc{ {0, 0}, ctx.swapchain_extent };
        vkCmdSetScissor(fd.command_buffer, 0, 1, &sc);

        VkDeviceSize offsets[] = { 0 };
        vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vertex_buffer, offsets);
        vkCmdBindIndexBuffer(fd.command_buffer, index_buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, main_pipeline);

        uint32_t model_slot = 0;

        // ---------- 1. Отрисовка объектов сцены с выбором материала и динамическим сдвигом ----------
        for (size_t i = 0; i < scene_objects.size() && model_slot < MAX_MODELS; ++i, ++model_slot) {
            int mat_idx = std::clamp(scene_objects[i].material_index, 0, (int)materials.size() - 1);
            VkDescriptorSet descriptor_sets[] = { scene_set, materials[mat_idx].descriptor_set };
            uint32_t dynamic_offset = static_cast<uint32_t>(model_slot * vk_buffer_model_uniforms_sizeof);

            vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                pipeline_layout, 0, 2, descriptor_sets, 1, &dynamic_offset);

            PipelineConsts consts{};
            consts.nontrivial_sampling = scene_objects[i].nontrivial_sampling ? 1u : 0u;
            consts.time = current_time_sec;
            consts.emissive_strength = emissive_strength;
            consts.wave_strength = wave_strength;

            vkCmdPushConstants(fd.command_buffer, pipeline_layout,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                0, sizeof(PipelineConsts), &consts);

            vkCmdDrawIndexed(fd.command_buffer, (uint32_t)indices.size(), 1, 0, 0, 0);
        }

        // ---------- 2. Визуализация источников света (через динамический UBO) ----------
        if (show_light_viz) {
            PipelineConsts consts{ 0u, current_time_sec, 0.0f, 0.0f };
            vkCmdPushConstants(fd.command_buffer, pipeline_layout,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                0, sizeof(PipelineConsts), &consts);

            VkDescriptorSet viz_sets[] = { scene_set, materials[0].descriptor_set };

            size_t total_light_cubes = point_lights.size() + spot_lights.size();
            for (size_t i = 0; i < total_light_cubes && model_slot < MAX_MODELS; ++i, ++model_slot) {
                uint32_t dynamic_offset = static_cast<uint32_t>(model_slot * vk_buffer_model_uniforms_sizeof);
                vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    pipeline_layout, 0, 2, viz_sets, 1, &dynamic_offset);
                vkCmdDrawIndexed(fd.command_buffer, (uint32_t)indices.size(), 1, 0, 0, 0);
            }

            if (line_vertices_count > 0 && model_slot < MAX_MODELS) {
                VkDeviceSize line_offset = 0;
                uint32_t dynamic_offset = static_cast<uint32_t>(model_slot * vk_buffer_model_uniforms_sizeof);
                vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, line_pipeline);
                vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &line_buffer, &line_offset);
                vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    pipeline_layout, 0, 2, viz_sets, 1, &dynamic_offset);
                vkCmdDraw(fd.command_buffer, (uint32_t)line_vertices_count, 1, 0, 0);
            }
        }

        vkCmdEndRenderPass(fd.command_buffer);
        vkEndCommandBuffer(fd.command_buffer);
    }

} // namespace application