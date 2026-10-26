// =============================================================
// application.cpp — Лабораторная работа №2: Камера и освещение
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

#include <vk_mem_alloc.h>

namespace application {

    namespace {
        // ============================================================
        // 1. ГЕОМЕТРИЯ КУБА С НОРМАЛЯМИ (24 вершины — по 4 на грань)
        // ============================================================
        struct Vertex {
            float position[3];
            float normal[3];
        };

        // Каждая грань имеет свою нормаль. 4 вершины на грань × 6 граней = 24.
        const std::vector<Vertex> vertices = {
            // Передняя грань (+Z)
            {{-1,-1, 1}, { 0, 0, 1}}, {{ 1,-1, 1}, { 0, 0, 1}},
            {{ 1, 1, 1}, { 0, 0, 1}}, {{-1, 1, 1}, { 0, 0, 1}},
            // Задняя (-Z)
            {{ 1,-1,-1}, { 0, 0,-1}}, {{-1,-1,-1}, { 0, 0,-1}},
            {{-1, 1,-1}, { 0, 0,-1}}, {{ 1, 1,-1}, { 0, 0,-1}},
            // Правая (+X)
            {{ 1,-1, 1}, { 1, 0, 0}}, {{ 1,-1,-1}, { 1, 0, 0}},
            {{ 1, 1,-1}, { 1, 0, 0}}, {{ 1, 1, 1}, { 1, 0, 0}},
            // Левая (-X)
            {{-1,-1,-1}, {-1, 0, 0}}, {{-1,-1, 1}, {-1, 0, 0}},
            {{-1, 1, 1}, {-1, 0, 0}}, {{-1, 1,-1}, {-1, 0, 0}},
            // Верхняя (+Y)
            {{-1, 1, 1}, { 0, 1, 0}}, {{ 1, 1, 1}, { 0, 1, 0}},
            {{ 1, 1,-1}, { 0, 1, 0}}, {{-1, 1,-1}, { 0, 1, 0}},
            // Нижняя (-Y)
            {{-1,-1,-1}, { 0,-1, 0}}, {{ 1,-1,-1}, { 0,-1, 0}},
            {{ 1,-1, 1}, { 0,-1, 0}}, {{-1,-1, 1}, { 0,-1, 0}},
        };

        const std::vector<uint32_t> indices = {
            0,1,2, 2,3,0,           // передняя
            4,5,6, 6,7,4,           // задняя
            8,9,10, 10,11,8,        // правая
            12,13,14, 14,15,12,     // левая
            16,17,18, 18,19,16,     // верхняя
            20,21,22, 22,23,20,     // нижняя
        };

        // ============================================================
        // 2. СТРУКТУРЫ ДАННЫХ (совпадают с GLSL по выравниванию)
        // ============================================================
        struct SceneUniforms {
            float view_projection[16];              // 64 B
            float camera_position[4];               // 16 B
            float directional_light_direction[4];   // 16 B
            float directional_light_color[4];       // 16 B
            float ambient_light[4];                 // 16 B
            uint32_t point_light_count;             // 4 B
            uint32_t spot_light_count;              // 4 B
            uint32_t _pad[2];                       // 8 B → итого 144 B
        };

        struct ModelUniforms {
            float model[16];        // 64 B
            float albedo[4];        // 16 B
            float specular[4];      // 16 B (rgb + shininess в .a)
        };

        struct PointLight {
            float position[4];      // xyz + pad
            float color[4];         // rgb + pad
        };  // 32 B

        struct SpotLight {
            float position[4];      // xyz + pad
            float direction[4];     // xyz + pad
            float color[4];         // rgb + pad
            float angles[4];        // x=inner, y=outer
        };  // 64 B

        struct PushConstants {
            float model[16];        // 64 B
            float color[4];         // 16 B
            uint32_t flags;         // 4 B
            uint32_t _pad[3];       // 12 B → выравнивание до 16
        };  // 96 B

        // ============================================================
        // 3. GUI-СОСТОЯНИЕ
        // ============================================================
        // --- Камера ---
        float camera_position[3] = { 0.0f, 2.0f, 6.0f };
        float camera_yaw = 0.0f;   // радианы (поворот влево-вправо)
        float camera_pitch = -0.2f;  // радианы (вверх-вниз)
        float camera_fov_deg = 60.0f;

        // --- Направленный источник ---
        float dir_light_direction[3] = { -0.5f, -1.0f, -0.3f };
        float dir_light_color[3] = { 1.0f,  1.0f,  1.0f };

        // --- Ambient ---
        float ambient_color[3] = { 0.1f, 0.1f, 0.15f };

        // --- Материал ---
        float material_albedo[3] = { 0.85f, 0.2f, 0.2f };
        float material_specular[3] = { 1.0f, 1.0f, 1.0f };
        float material_shininess = 64.0f;

        // --- Массивы источников ---
        std::vector<PointLight> point_lights = {
            {{ 2.0f, 2.0f, 2.0f, 0.0f}, {1.0f, 0.5f, 0.2f, 1.0f}},
        };
        std::vector<SpotLight> spot_lights = {
            {{0.0f, 3.0f, 3.0f, 0.0f}, {0.0f,-1.0f,-0.5f,0.0f}, {0.3f,0.6f,1.0f,1.0f}, {20.0f,35.0f,0.0f,0.0f}},
        };

        // --- Настройки визуализации ---
        bool show_light_viz = true;

        // ============================================================
        // 4. VULKAN-ОБЪЕКТЫ
        // ============================================================
        // --- Буферы геометрии ---
        VkBuffer      vertex_buffer = VK_NULL_HANDLE;
        VmaAllocation vertex_buffer_alloc = VK_NULL_HANDLE;
        VkBuffer      index_buffer = VK_NULL_HANDLE;
        VmaAllocation index_buffer_alloc = VK_NULL_HANDLE;

        // --- Scene UBO (set 0, binding 0) ---
        VkBuffer      scene_ubo = VK_NULL_HANDLE;
        VmaAllocation scene_ubo_alloc = VK_NULL_HANDLE;
        void* scene_ubo_mapped = nullptr;

        // --- Model UBO (set 1, binding 0) ---
        VkBuffer      model_ubo = VK_NULL_HANDLE;
        VmaAllocation model_ubo_alloc = VK_NULL_HANDLE;
        void* model_ubo_mapped = nullptr;

        // --- Storage buffers для источников (set 0, binding 1/2) ---
        VkBuffer      point_lights_ssbo = VK_NULL_HANDLE;
        VmaAllocation point_lights_ssbo_alloc = VK_NULL_HANDLE;
        void* point_lights_ssbo_mapped = nullptr;

        VkBuffer      spot_lights_ssbo = VK_NULL_HANDLE;
        VmaAllocation spot_lights_ssbo_alloc = VK_NULL_HANDLE;
        void* spot_lights_ssbo_mapped = nullptr;

        // --- Максимальное количество источников (для преаллокации) ---
        constexpr size_t MAX_POINT_LIGHTS = 32;
        constexpr size_t MAX_SPOT_LIGHTS = 16;

        // --- Вершинный буфер для линий (конусов прожекторов) ---
        struct LineVertex { float position[3]; };
        constexpr size_t MAX_LINE_VERTICES = 4096;
        VkBuffer      line_buffer = VK_NULL_HANDLE;
        VmaAllocation line_buffer_alloc = VK_NULL_HANDLE;
        void* line_buffer_mapped = nullptr;
        size_t        line_vertices_count = 0;

        // --- Дескрипторы ---
        VkDescriptorSetLayout scene_set_layout = VK_NULL_HANDLE;
        VkDescriptorSetLayout model_set_layout = VK_NULL_HANDLE;
        VkDescriptorPool      descriptor_pool = VK_NULL_HANDLE;
        VkDescriptorSet       scene_set = VK_NULL_HANDLE;
        VkDescriptorSet       model_set = VK_NULL_HANDLE;

        // --- Пайплайны ---
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        VkPipeline       main_pipeline = VK_NULL_HANDLE;  // TRIANGLE_LIST
        VkPipeline       line_pipeline = VK_NULL_HANDLE;  // LINE_LIST

        // ============================================================
        // 5. МИНИ-МАТЕМАТИКА
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

        Mat4 perspective(float fovy, float aspect, float zn, float zf) {
            Mat4 r{};
            float t = std::tan(fovy * 0.5f);
            r.m[0] = 1.0f / (aspect * t);
            r.m[5] = -1.0f / t;              // Y-flip для Vulkan
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

        // ============================================================
        // 6. ВСПОМОГАТЕЛЬНОЕ
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

        // ---------- 1. Вершинный буфер ----------
        VkBufferCreateInfo vb_info{};
        vb_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        vb_info.size = vertices.size() * sizeof(Vertex);
        vb_info.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        vb_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo alloc_info{};
        alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
        alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
            | VMA_ALLOCATION_CREATE_MAPPED_BIT;

        VmaAllocationInfo vb_ai{};
        if (vmaCreateBuffer(ctx.allocator, &vb_info, &alloc_info,
            &vertex_buffer, &vertex_buffer_alloc, &vb_ai) != VK_SUCCESS) {
            std::cerr << "[app] vertex buffer failed\n"; return false;
        }
        std::memcpy(vb_ai.pMappedData, vertices.data(), vb_info.size);

        // ---------- 2. Индексный буфер ----------
        VkBufferCreateInfo ib_info = vb_info;
        ib_info.size = indices.size() * sizeof(uint32_t);
        ib_info.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;

        VmaAllocationInfo ib_ai{};
        if (vmaCreateBuffer(ctx.allocator, &ib_info, &alloc_info,
            &index_buffer, &index_buffer_alloc, &ib_ai) != VK_SUCCESS) {
            std::cerr << "[app] index buffer failed\n"; return false;
        }
        std::memcpy(ib_ai.pMappedData, indices.data(), ib_info.size);

        // ---------- 3. Scene UBO ----------
        if (!createMappedBuffer(sizeof(SceneUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            scene_ubo, scene_ubo_alloc, &scene_ubo_mapped)) {
            std::cerr << "[app] scene ubo failed\n"; return false;
        }

        // ---------- 4. Model UBO ----------
        if (!createMappedBuffer(sizeof(ModelUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            model_ubo, model_ubo_alloc, &model_ubo_mapped)) {
            std::cerr << "[app] model ubo failed\n"; return false;
        }

        // ---------- 5. Storage buffers для источников ----------
        if (!createMappedBuffer(sizeof(PointLight) * MAX_POINT_LIGHTS,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            point_lights_ssbo, point_lights_ssbo_alloc,
            &point_lights_ssbo_mapped)) {
            std::cerr << "[app] point lights ssbo failed\n"; return false;
        }
        if (!createMappedBuffer(sizeof(SpotLight) * MAX_SPOT_LIGHTS,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            spot_lights_ssbo, spot_lights_ssbo_alloc,
            &spot_lights_ssbo_mapped)) {
            std::cerr << "[app] spot lights ssbo failed\n"; return false;
        }

        // ---------- 6. Вершинный буфер для линий конусов ----------
        if (!createMappedBuffer(sizeof(LineVertex) * MAX_LINE_VERTICES,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            line_buffer, line_buffer_alloc,
            &line_buffer_mapped)) {
            std::cerr << "[app] line buffer failed\n"; return false;
        }

        // 7. DESCRIPTOR SET LAYOUT (два набора)
        VkDescriptorSetLayoutBinding scene_bindings[3]{};

        scene_bindings[0].binding = 0;
        scene_bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        scene_bindings[0].descriptorCount = 1;
        scene_bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        scene_bindings[1].binding = 1;
        scene_bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        scene_bindings[1].descriptorCount = 1;
        scene_bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        scene_bindings[2].binding = 2;
        scene_bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        scene_bindings[2].descriptorCount = 1;
        scene_bindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo scene_dsl{};
        scene_dsl.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        scene_dsl.bindingCount = 3;
        scene_dsl.pBindings = scene_bindings;
        if (vkCreateDescriptorSetLayout(device, &scene_dsl, nullptr, &scene_set_layout) != VK_SUCCESS) {
            std::cerr << "[app] scene dsl failed\n"; return false;
        }

        // Set 1: Model (UBO)
        VkDescriptorSetLayoutBinding model_binding{};
        model_binding.binding = 0;
        model_binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        model_binding.descriptorCount = 1;
        model_binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo model_dsl{};
        model_dsl.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        model_dsl.bindingCount = 1;
        model_dsl.pBindings = &model_binding;
        if (vkCreateDescriptorSetLayout(device, &model_dsl, nullptr, &model_set_layout) != VK_SUCCESS) {
            std::cerr << "[app] model dsl failed\n"; return false;
        }

        // 8. DESCRIPTOR POOL + SETS
        VkDescriptorPoolSize pool_sizes[2]{};
        pool_sizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        pool_sizes[0].descriptorCount = 8;
        pool_sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pool_sizes[1].descriptorCount = 4;

        VkDescriptorPoolCreateInfo dp{};
        dp.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dp.maxSets = 4;
        dp.poolSizeCount = 2;
        dp.pPoolSizes = pool_sizes;

        if (vkCreateDescriptorPool(device, &dp, nullptr, &descriptor_pool) != VK_SUCCESS) {
            std::cerr << "[app] descriptor pool failed\n"; return false;
        }

        // Аллоцируем scene_set
        VkDescriptorSetAllocateInfo ds_alloc{};
        ds_alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ds_alloc.descriptorPool = descriptor_pool;
        ds_alloc.descriptorSetCount = 1;
        ds_alloc.pSetLayouts = &scene_set_layout;
        vkAllocateDescriptorSets(device, &ds_alloc, &scene_set);

        // Аллоцируем model_set
        ds_alloc.pSetLayouts = &model_set_layout;
        vkAllocateDescriptorSets(device, &ds_alloc, &model_set);

        // Связываем scene_set: UBO + 2 SSBO Положи этот буфер в scene_set в слот binding = 1
        VkDescriptorBufferInfo scene_info{};
        scene_info.buffer = scene_ubo;
        scene_info.range = sizeof(SceneUniforms);

        VkDescriptorBufferInfo point_info{};
        point_info.buffer = point_lights_ssbo;
        point_info.range = sizeof(PointLight) * MAX_POINT_LIGHTS;

        VkDescriptorBufferInfo spot_info{};
        spot_info.buffer = spot_lights_ssbo;
        spot_info.range = sizeof(SpotLight) * MAX_SPOT_LIGHTS;

        VkWriteDescriptorSet writes[3]{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = scene_set;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[0].pBufferInfo = &scene_info;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = scene_set;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].pBufferInfo = &point_info;

        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet = scene_set;
        writes[2].dstBinding = 2;
        writes[2].descriptorCount = 1;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[2].pBufferInfo = &spot_info;

        vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);

        // Связываем model_set: UBO
        VkDescriptorBufferInfo model_info{};
        model_info.buffer = model_ubo;
        model_info.range = sizeof(ModelUniforms);

        VkWriteDescriptorSet mwrite{};
        mwrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        mwrite.dstSet = model_set;
        mwrite.dstBinding = 0;
        mwrite.descriptorCount = 1;
        mwrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        mwrite.pBufferInfo = &model_info;
        vkUpdateDescriptorSets(device, 1, &mwrite, 0, nullptr);

        // ============================================================
        // 9. PIPELINE LAYOUT (два set-а + push constants)
        // ============================================================
        VkDescriptorSetLayout all_sets[] = { scene_set_layout, model_set_layout };

        VkPushConstantRange pc_range{};
        pc_range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        pc_range.offset = 0;
        pc_range.size = sizeof(PushConstants);

        VkPipelineLayoutCreateInfo pl_info{};
        pl_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl_info.setLayoutCount = 2;
        pl_info.pSetLayouts = all_sets;
        pl_info.pushConstantRangeCount = 1;
        pl_info.pPushConstantRanges = &pc_range;

        if (vkCreatePipelineLayout(device, &pl_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
            std::cerr << "[app] pipeline layout failed\n"; return false;
        }

        // ============================================================
        // 10. SHADER MODULES
        // ============================================================
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

        // ============================================================
        // 11. ОБЩИЕ СОСТОЯНИЯ ПАЙПЛАЙНА
        // ============================================================
        // Layout для куба: position + normal
        VkVertexInputBindingDescription vib{};
        vib.binding = 0;
        vib.stride = sizeof(Vertex);
        vib.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        VkVertexInputAttributeDescription via[2]{};
        via[0].location = 0;
        via[0].binding = 0;
        via[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        via[0].offset = offsetof(Vertex, position);

        via[1].location = 1;
        via[1].binding = 0;
        via[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        via[1].offset = offsetof(Vertex, normal);

        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vi.vertexBindingDescriptionCount = 1;
        vi.pVertexBindingDescriptions = &vib;
        vi.vertexAttributeDescriptionCount = 2;
        vi.pVertexAttributeDescriptions = via;

        VkPipelineViewportStateCreateInfo vp{};
        vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vp.viewportCount = 1;
        vp.scissorCount = 1;

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

        // ============================================================
        // 12. MAIN PIPELINE (TRIANGLE_LIST) — куб и кубики источников
        // ============================================================
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

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipe, nullptr, &main_pipeline) != VK_SUCCESS) {
            std::cerr << "[app] main pipeline failed\n"; return false;
        }

        // ============================================================
        // 13. LINE PIPELINE (LINE_LIST) — конусы прожекторов
        // ============================================================
        ia.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
        rs.cullMode = VK_CULL_MODE_NONE;

        // У линий только позиция (12 байт)
        VkVertexInputBindingDescription line_vib{};
        line_vib.binding = 0;
        line_vib.stride = sizeof(LineVertex);
        line_vib.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        VkVertexInputAttributeDescription line_via{};
        line_via.location = 0;
        line_via.binding = 0;
        line_via.format = VK_FORMAT_R32G32B32_SFLOAT;
        line_via.offset = 0;

        VkPipelineVertexInputStateCreateInfo line_vi{};
        line_vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        line_vi.vertexBindingDescriptionCount = 1;
        line_vi.pVertexBindingDescriptions = &line_vib;
        line_vi.vertexAttributeDescriptionCount = 1;
        line_vi.pVertexAttributeDescriptions = &line_via;

        pipe.pVertexInputState = &line_vi;

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipe, nullptr, &line_pipeline) != VK_SUCCESS) {
            std::cerr << "[app] line pipeline failed\n"; return false;
        }

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

        vmaDestroyBuffer(ctx.allocator, line_buffer, line_buffer_alloc);
        vmaDestroyBuffer(ctx.allocator, spot_lights_ssbo, spot_lights_ssbo_alloc);
        vmaDestroyBuffer(ctx.allocator, point_lights_ssbo, point_lights_ssbo_alloc);
        vmaDestroyBuffer(ctx.allocator, model_ubo, model_ubo_alloc);
        vmaDestroyBuffer(ctx.allocator, scene_ubo, scene_ubo_alloc);
        vmaDestroyBuffer(ctx.allocator, index_buffer, index_buffer_alloc);
        vmaDestroyBuffer(ctx.allocator, vertex_buffer, vertex_buffer_alloc);
    }

    // ================================================================
    // UPDATE
    // ================================================================
    void update(double time_delta) {
        auto& ctx = graphics::internal::context;
        const float DEG2RAD = 3.14159265358979323846f / 180.0f;
        (void)time_delta;

        // ---------- GUI ----------
        ImGui::Begin("Scene Settings");

        if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::DragFloat3("Position", camera_position, 0.1f);
            ImGui::SliderAngle("Yaw", &camera_yaw, -180.0f, 180.0f);
            ImGui::SliderAngle("Pitch", &camera_pitch, -89.0f, 89.0f);
            ImGui::SliderFloat("FOV", &camera_fov_deg, 20.0f, 120.0f);
        }

        if (ImGui::CollapsingHeader("Directional Light", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::DragFloat3("Direction##dir", dir_light_direction, 0.05f);
            ImGui::ColorEdit3("Color##dir", dir_light_color);
        }

        if (ImGui::CollapsingHeader("Ambient", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::ColorEdit3("Ambient color", ambient_color);
        }

        if (ImGui::CollapsingHeader("Material", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::ColorEdit3("Albedo", material_albedo);
            ImGui::ColorEdit3("Specular", material_specular);
            ImGui::SliderFloat("Shininess", &material_shininess, 1.0f, 256.0f);
        }

        if (ImGui::CollapsingHeader("Point Lights", ImGuiTreeNodeFlags_DefaultOpen)) {
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

        if (ImGui::CollapsingHeader("Spot Lights", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::Button("Add Spot Light") && spot_lights.size() < MAX_SPOT_LIGHTS) {
                spot_lights.push_back({
                    {0, 3, 0, 0}, {0, -1, 0, 0}, {1, 1, 1, 1}, {20, 35, 0, 0}
                    });
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

        // ---------- Матрица view-projection ----------
        Mat4 view = computeView();
        float aspect = (float)ctx.swapchain_extent.width / (float)ctx.swapchain_extent.height;
        Mat4 proj = perspective(camera_fov_deg * DEG2RAD, aspect, 0.1f, 100.0f);
        Mat4 view_proj = multiply(proj, view);

        // ---------- Scene UBO ----------
        SceneUniforms su{};
        std::memcpy(su.view_projection, view_proj.m, sizeof(view_proj.m));
        su.camera_position[0] = camera_position[0];
        su.camera_position[1] = camera_position[1];
        su.camera_position[2] = camera_position[2];
        su.camera_position[3] = 1.0f;

        su.directional_light_direction[0] = dir_light_direction[0];
        su.directional_light_direction[1] = dir_light_direction[1];
        su.directional_light_direction[2] = dir_light_direction[2];
        su.directional_light_direction[3] = 0.0f;

        std::memcpy(su.directional_light_color, dir_light_color, sizeof(dir_light_color));
        su.directional_light_color[3] = 1.0f;

        std::memcpy(su.ambient_light, ambient_color, sizeof(ambient_color));
        su.ambient_light[3] = 1.0f;

        su.point_light_count = (uint32_t)point_lights.size();
        su.spot_light_count = (uint32_t)spot_lights.size();

        std::memcpy(scene_ubo_mapped, &su, sizeof(su));

        // ---------- Model UBO ----------
        ModelUniforms mu{};
        Mat4 identity_model = identity();
        std::memcpy(mu.model, identity_model.m, sizeof(identity_model.m));
        std::memcpy(mu.albedo, material_albedo, sizeof(material_albedo));
        mu.albedo[3] = 1.0f;
        std::memcpy(mu.specular, material_specular, sizeof(material_specular));
        mu.specular[3] = material_shininess;
        std::memcpy(model_ubo_mapped, &mu, sizeof(mu));

        // ---------- Storage buffers ----------
        if (!point_lights.empty())
            std::memcpy(point_lights_ssbo_mapped, point_lights.data(),
                point_lights.size() * sizeof(PointLight));
        if (!spot_lights.empty())
            std::memcpy(spot_lights_ssbo_mapped, spot_lights.data(),
                spot_lights.size() * sizeof(SpotLight));

        // ---------- Буфер линий для конусов прожекторов ----------
        line_vertices_count = 0;
        if (show_light_viz) {
            LineVertex* verts = static_cast<LineVertex*>(line_buffer_mapped);
            constexpr int SEGMENTS = 24;
            constexpr float CONE_LENGTH = 2.0f;
            constexpr float PI = 3.14159265358979323846f;

            for (const auto& sl : spot_lights) {
                // Проверяем, что хватит места под: 24 луча * 2 + 24 окружности * 2 = 96 вершин
                if (line_vertices_count + SEGMENTS * 4 > MAX_LINE_VERTICES) break;

                float ox = sl.position[0], oy = sl.position[1], oz = sl.position[2];
                float dx = sl.direction[0], dy = sl.direction[1], dz = sl.direction[2];
                float dl = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (dl < 1e-5f) continue;
                dx /= dl; dy /= dl; dz /= dl;

                // Вектор u — перпендикуляр к оси конуса
                float ux, uy, uz;
                if (std::fabs(dx) < 0.9f) {
                    float cx = 1.0f, cy = 0.0f, cz = 0.0f;
                    ux = dy * cz - dz * cy;
                    uy = dz * cx - dx * cz;
                    uz = dx * cy - dy * cx;
                }
                else {
                    float cx = 0.0f, cy = 1.0f, cz = 0.0f;
                    ux = dy * cz - dz * cy;
                    uy = dz * cx - dx * cz;
                    uz = dx * cy - dy * cx;
                }
                float ul = std::sqrt(ux * ux + uy * uy + uz * uz);
                ux /= ul; uy /= ul; uz /= ul;

                // Вектор v = cross(d, u)
                float vx = dy * uz - dz * uy;
                float vy = dz * ux - dx * uz;
                float vz = dx * uy - dy * ux;

                // Радиус основания конуса
                float outer_deg = sl.angles[1];
                float radius = CONE_LENGTH * std::tan(outer_deg * PI / 180.0f);

                // Центр основания
                float bx = ox + dx * CONE_LENGTH;
                float by = oy + dy * CONE_LENGTH;
                float bz = oz + dz * CONE_LENGTH;

                // Точки окружности
                float circle_pts[24 * 3];
                for (int i = 0; i < SEGMENTS; ++i) {
                    float t = (float)i / (float)SEGMENTS * 2.0f * PI;
                    float c = std::cos(t), s = std::sin(t);
                    circle_pts[i * 3 + 0] = bx + radius * (c * ux + s * vx);
                    circle_pts[i * 3 + 1] = by + radius * (c * uy + s * vy);
                    circle_pts[i * 3 + 2] = bz + radius * (c * uz + s * vz);
                }

                // Лучи от вершины к точкам окружности
                for (int i = 0; i < SEGMENTS; ++i) {
                    verts[line_vertices_count++] = { { ox, oy, oz } };
                    verts[line_vertices_count++] = { { circle_pts[i * 3 + 0],
                                                       circle_pts[i * 3 + 1],
                                                       circle_pts[i * 3 + 2] } };
                }

                // Окружность основания
                for (int i = 0; i < SEGMENTS; ++i) {
                    int j = (i + 1) % SEGMENTS;
                    verts[line_vertices_count++] = { { circle_pts[i * 3 + 0],
                                                       circle_pts[i * 3 + 1],
                                                       circle_pts[i * 3 + 2] } };
                    verts[line_vertices_count++] = { { circle_pts[j * 3 + 0],
                                                       circle_pts[j * 3 + 1],
                                                       circle_pts[j * 3 + 2] } };
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
        clear_values[0].color = { {0.05f, 0.05f, 0.08f, 1.0f} };
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

        VkViewport vp{};
        vp.width = (float)ctx.swapchain_extent.width;
        vp.height = (float)ctx.swapchain_extent.height;
        vp.minDepth = 0.0f;
        vp.maxDepth = 1.0f;
        vkCmdSetViewport(fd.command_buffer, 0, 1, &vp);

        VkRect2D sc{};
        sc.extent = ctx.swapchain_extent;
        vkCmdSetScissor(fd.command_buffer, 0, 1, &sc);

        // Общие буферы
        VkDeviceSize offsets[] = { 0 };
        vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vertex_buffer, offsets);
        vkCmdBindIndexBuffer(fd.command_buffer, index_buffer, 0, VK_INDEX_TYPE_UINT32);

        // Дескрипторные наборы
        VkDescriptorSet sets[] = { scene_set, model_set };
        vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipeline_layout, 0, 2, sets, 0, nullptr);

        // ---------- 1. Основной куб ----------
        PushConstants pc{};
        Mat4 ident = identity();
        std::memcpy(pc.model, ident.m, sizeof(ident.m));
        pc.flags = 0;  // lit

        vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, main_pipeline);
        vkCmdPushConstants(fd.command_buffer, pipeline_layout,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            0, sizeof(pc), &pc);
        vkCmdDrawIndexed(fd.command_buffer, (uint32_t)indices.size(), 1, 0, 0, 0);

        // ---------- 2. Визуализация источников ----------
        if (show_light_viz) {
            vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, main_pipeline);

            // Точечные источники — кубики цвета
            for (const auto& pl : point_lights) {
                Mat4 model = multiply(
                    translate(pl.position[0], pl.position[1], pl.position[2]),
                    scaleM(0.15f, 0.15f, 0.15f)
                );
                std::memcpy(pc.model, model.m, sizeof(model.m));
                pc.color[0] = pl.color[0];
                pc.color[1] = pl.color[1];
                pc.color[2] = pl.color[2];
                pc.color[3] = 1.0f;
                pc.flags = 1;  // unlit

                vkCmdPushConstants(fd.command_buffer, pipeline_layout,
                    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                    0, sizeof(pc), &pc);
                vkCmdDrawIndexed(fd.command_buffer, (uint32_t)indices.size(), 1, 0, 0, 0);
            }

            // Прожекторы — кубики цвета
            for (const auto& sl : spot_lights) {
                Mat4 model = multiply(
                    translate(sl.position[0], sl.position[1], sl.position[2]),
                    scaleM(0.15f, 0.15f, 0.15f)
                );
                std::memcpy(pc.model, model.m, sizeof(model.m));
                pc.color[0] = sl.color[0];
                pc.color[1] = sl.color[1];
                pc.color[2] = sl.color[2];
                pc.color[3] = 1.0f;
                pc.flags = 1;

                vkCmdPushConstants(fd.command_buffer, pipeline_layout,
                    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                    0, sizeof(pc), &pc);
                vkCmdDrawIndexed(fd.command_buffer, (uint32_t)indices.size(), 1, 0, 0, 0);
            }

            // Конусы прожекторов — белые линии
            if (line_vertices_count > 0) {
                VkDeviceSize line_offset = 0;
                vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, line_pipeline);
                vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &line_buffer, &line_offset);
                vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    pipeline_layout, 0, 2, sets, 0, nullptr);

                Mat4 ident2 = identity();
                std::memcpy(pc.model, ident2.m, sizeof(ident2.m));
                pc.color[0] = 1.0f; pc.color[1] = 1.0f; pc.color[2] = 1.0f; pc.color[3] = 1.0f;
                pc.flags = 1;
                vkCmdPushConstants(fd.command_buffer, pipeline_layout,
                    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                    0, sizeof(pc), &pc);

                vkCmdDraw(fd.command_buffer, (uint32_t)line_vertices_count, 1, 0, 0);
            }
        }

        vkCmdEndRenderPass(fd.command_buffer);
        vkEndCommandBuffer(fd.command_buffer);
    }

} // namespace application