// Подключение заголовков
#include "application.hpp"          // заголовок с объявлениями initialize/shutdown/update/render
#include <imgui.h>                  // Библиотека интерфейса ImGui (для панели с ползунками)
#include <vector>                   // std::vector — динамические массивы вершин/индексов
#include <cmath>                    // std::sin, std::cos, std::tan, std::sqrt — математика
#include <cstring>                  // std::memcpy — копирование данных в память GPU
#include <cstddef>                  // offsetof — смещение поля внутри структуры
#include <string>                   // std::string для путей к шейдерам
#include <fstream>                  // Чтение .spv файлов с диска
#include <iostream>                 // std::cerr — вывод ошибок

#include <vk_mem_alloc.h>           // Vulkan Memory Allocator — упрощает работу с памятью GPU

// Всё в namespace application, чтобы не засорять глобальное пространство имён
namespace application {

// Анонимный namespace — всё внутри видно только в этом файле (аналог static)
namespace {
    // 1.геометрия куба
    // Описание одной вершины
    struct Vertex {
        float position[3];          // x, y, z
    };

    // 8 вершин куба (углы единичного куба от -1 до +1)
    // Вершины пронумерованы: 0-3 — передняя грань, 4-7 — задняя
    const std::vector<Vertex> vertices = {
        {{-1.0f, -1.0f,  1.0f}},    // 0: левый-нижний-передний
        {{ 1.0f, -1.0f,  1.0f}},    // 1: правый-нижний-передний
        {{ 1.0f,  1.0f,  1.0f}},    // 2: правый-верхний-передний
        {{-1.0f,  1.0f,  1.0f}},    // 3: левый-верхний-передний
        {{-1.0f, -1.0f, -1.0f}},    // 4: левый-нижний-задний
        {{ 1.0f, -1.0f, -1.0f}},    // 5: правый-нижний-задний
        {{ 1.0f,  1.0f, -1.0f}},    // 6: правый-верхний-задний
        {{-1.0f,  1.0f, -1.0f}}     // 7: левый-верхний-задний
    };

    // Индексы — порядок обхода вершин для рисования 12 треугольников (6 граней × 2 треугольника)
    // Каждое число — номер вершины из массива vertices. Так экономится память:
    // вместо 36 вершин храним 8 + 36 индексов.
    const std::vector<uint32_t> indices = {
        0, 1, 2, 2, 3, 0,           // Передняя грань (z = +1)
        1, 5, 6, 6, 2, 1,           // Правая грань  (x = +1)
        5, 4, 7, 7, 6, 5,           // Задняя грань  (z = -1)
        4, 0, 3, 3, 7, 4,           // Левая грань   (x = -1)
        3, 2, 6, 6, 7, 3,           // Верхняя грань (y = +1)
        4, 5, 1, 1, 0, 4            // Нижняя грань  (y = -1)
    };
    // 2.Переменные (GUI)
    // все переменные напрямую связаны с ползунками ImGui, Изменение ползунка → изменение переменной → изменение матрицы Model.
    bool  is_perspective = true;                // Перспектива (true) или ортографика (false)
    float pos[3]   = {0.0f, 0.0f, 0.0f};        // Позиция куба
    float rot[3]   = {0.0f, 0.0f, 0.0f};        // Углы поворота (в градусах)
    float scale[3] = {1.0f, 1.0f, 1.0f};        // Масштаб по осям
    float color[4] = {1.0f, 1.0f, 1.0f, 1.0f};  // RGBA-цвет куба

    bool  is_playing   = false;                 // Флаг анимации
    float anim_speed   = 1.0f;                  // Скорость анимации
    float anim_radius  = 2.0f;                  // Радиус траектории
    float current_time = 0.0f;                  // Накопленное время анимации

    //3.Vulkan объекты (создаются в initialize, уничтожаются в shutdown)
    // VkBuffer — область памяти на GPU. VmaAllocation — структура VMA, описывающая эту память
    VkBuffer      vertex_buffer       = VK_NULL_HANDLE;    // Буфер с координатами 8 вершин
    VmaAllocation vertex_buffer_alloc = VK_NULL_HANDLE;    // Аллокация под вершинный буфер

    VkBuffer      index_buffer        = VK_NULL_HANDLE;    // Буфер с 36 индексами
    VmaAllocation index_buffer_alloc  = VK_NULL_HANDLE;    // Аллокация под индексный буфер

    VkBuffer      uniform_buffer        = VK_NULL_HANDLE;  // Буфер с MVP-матрицей и цветом
    VmaAllocation uniform_buffer_alloc  = VK_NULL_HANDLE;
    void*         uniform_buffer_mapped = nullptr;         // Указатель на память CPU, куда пишем матрицу

    // Descriptor — передача данных в шейдер
    // Layout описывает какие ресурсы и в каких слотах, Pool — пул для выделения, Set —сам набор
    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    VkDescriptorPool      descriptor_pool       = VK_NULL_HANDLE;
    VkDescriptorSet       descriptor_set        = VK_NULL_HANDLE;

    // Pipeline Layout — связка дескрипторов и push-констант для пайплайна
    // Pipeline — скомпилированный граф операций GPU (шейдеры + состояния)
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline       pipeline        = VK_NULL_HANDLE;

    // математика(матрицы 4x4, column-major)
    // Column-major: элемент в строке r, столбце c лежит в m[c*4 + r]
    struct Mat4 { float m[16]; };

    Mat4 identity() {
        Mat4 r{};
        r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;   //Диагональ = 1
        return r;
    }

    // Умножение матриц: a * b (не коммутативно)
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

    //Матрица переноса: смещает точку на (x, y, z)
    Mat4 translate(float x, float y, float z) {
        Mat4 r = identity();
        r.m[12] = x; r.m[13] = y; r.m[14] = z;
        return r;
    }

    // Матрица масштаба: растягивает по осям
    Mat4 scaleM(float x, float y, float z) {
        Mat4 r = identity();
        r.m[0] = x; r.m[5] = y; r.m[10] = z;
        return r;
    }

    // Поворот вокруг оси X (в радианах), Стандартные формулы 2D-поворота в плоскости YZ.
    Mat4 rotX(float a) {
        Mat4 r = identity();
        float c = std::cos(a), s = std::sin(a);
        r.m[5] = c; r.m[6] = s; r.m[9] = -s; r.m[10] = c;
        return r;
    }
    // Поворот вокруг Y
    Mat4 rotY(float a) {
        Mat4 r = identity();
        float c = std::cos(a), s = std::sin(a);
        r.m[0] = c; r.m[2] = -s; r.m[8] = s; r.m[10] = c;
        return r;
    }
    // Поворот вокруг Z
    Mat4 rotZ(float a) {
        Mat4 r = identity();
        float c = std::cos(a), s = std::sin(a);
        r.m[0] = c; r.m[1] = s; r.m[4] = -s; r.m[5] = c;
        return r;
    }

    // Перспективная проекция. Отличия от OpenGL:
    //  - Vulkan использует Z ∈ [0, 1] (OpenGL: [-1, 1])
    //  - Vulkan имеет Y ось, направленную вниз - минус перед m[5]
    Mat4 perspective(float fovy, float aspect, float zn, float zf) {
        Mat4 r{};
        float t = std::tan(fovy * 0.5f);
        r.m[0]  = 1.0f / (aspect * t);       // масштаб по X
        r.m[5]  = -1.0f / t;                 // масштаб по Y
        r.m[10] = zf / (zn - zf);            // пересчёт z в [0, 1]
        r.m[11] = -1.0f;                     // W = -z
        r.m[14] = (zf * zn) / (zn - zf);
        return r;
    }

    // Ортографическая проекция (без учёта глубины) с Y флипом.
    Mat4 ortho(float l, float rr, float b, float t, float n, float f) {
        Mat4 r{};
        r.m[0]  = 2.0f / (rr - l);
        r.m[5]  = -2.0f / (t - b);           // Y флип
        r.m[10] = 1.0f / (n - f);
        r.m[12] = -(rr + l) / (rr - l);
        r.m[13] =  (t + b) / (t - b);
        r.m[14] =  n / (n - f);
        r.m[15] =  1.0f;
        return r;
    }

    // Матрица вида (камера)
    //  f = нормализованное направление взгляда (center - eye)
    //  s = нормализованное правое направление (f × up)
    //  u = истинный "верх" камеры (s × f)
    Mat4 lookAt(const float eye[3], const float center[3], const float up[3]) {
        float f[3] = {center[0]-eye[0], center[1]-eye[1], center[2]-eye[2]};
        float fl = std::sqrt(f[0]*f[0]+f[1]*f[1]+f[2]*f[2]);
        f[0]/=fl; f[1]/=fl; f[2]/=fl;

        float s[3] = {f[1]*up[2]-f[2]*up[1], f[2]*up[0]-f[0]*up[2], f[0]*up[1]-f[1]*up[0]};
        float sl = std::sqrt(s[0]*s[0]+s[1]*s[1]+s[2]*s[2]);
        s[0]/=sl; s[1]/=sl; s[2]/=sl;

        float u[3] = {s[1]*f[2]-s[2]*f[1], s[2]*f[0]-s[0]*f[2], s[0]*f[1]-s[1]*f[0]};

        Mat4 r = identity();
        r.m[0]=s[0]; r.m[1]=u[0]; r.m[2]=-f[0];
        r.m[4]=s[1]; r.m[5]=u[1]; r.m[6]=-f[1];
        r.m[8]=s[2]; r.m[9]=u[2]; r.m[10]=-f[2];
        r.m[12]=-(s[0]*eye[0]+s[1]*eye[1]+s[2]*eye[2]);
        r.m[13]=-(u[0]*eye[0]+u[1]*eye[1]+u[2]*eye[2]);
        r.m[14]= (f[0]*eye[0]+f[1]*eye[1]+f[2]*eye[2]);
        return r;
    }

    // 5.вспомогательные функии
    // Читаем файл целиком в вектор char. Нужно для загрузки .spv шейдеров
    std::vector<char> readFile(const std::string& path) {
        std::ifstream file(path, std::ios::ate | std::ios::binary);   //ate — сразу прыгаем в конец
        if (!file.is_open()) {
            std::cerr << "[app] Не удалось открыть файл: " << path << "\n";
            return {};
        }
        size_t size = static_cast<size_t>(file.tellg());    //размер = текущая позиция (конец)
        std::vector<char> buf(size);
        file.seekg(0);                                       // возвращаемся в начало
        file.read(buf.data(), size);
        return buf;
    }

    // айты шейдера в объект Vulkan — VkShaderModule
    VkShaderModule createShaderModule(const std::vector<char>& code) {
        VkShaderModuleCreateInfo info{};
        info.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        info.codeSize = code.size();
        info.pCode    = reinterpret_cast<const uint32_t*>(code.data());
        VkShaderModule m = VK_NULL_HANDLE;
        if (vkCreateShaderModule(graphics::internal::context.device, &info, nullptr, &m) != VK_SUCCESS) {
            std::cerr << "[app] Не удалось создать VkShaderModule\n";
            return VK_NULL_HANDLE;
        }
        return m;
    }
} // namespace

// initialize — создаём все ресурсы Vulkan при старте
bool initialize() {
    auto& ctx = graphics::internal::context;
    VkDevice device = ctx.device;

    //Вершинный буфер-
    VkBufferCreateInfo vb_info{};
    vb_info.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    vb_info.size        = vertices.size() * sizeof(Vertex);       // 8 вершин × 12 байт = 96 байт
    vb_info.usage       = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;      // назначение — вершинный
    vb_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;              // используется одним семейством очередей

    VmaAllocationCreateInfo alloc_info{};
    alloc_info.usage = VMA_MEMORY_USAGE_AUTO;                      // VMA сам подберёт тип памяти
    // HOST_ACCESS — память доступна CPU (для записи)
    // MAPPED — VMA сразу замапит её, дав нам указатель pMappedData
    alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                     | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo ai{};
    if (vmaCreateBuffer(ctx.allocator, &vb_info, &alloc_info,
                        &vertex_buffer, &vertex_buffer_alloc, &ai) != VK_SUCCESS) {
        std::cerr << "[app] vmaCreateBuffer(vertex) failed\n";
        return false;
    }
    // Копируем вершины в память GPU
    std::memcpy(ai.pMappedData, vertices.data(), vb_info.size);

    //Индексный буфер
    VkBufferCreateInfo ib_info{};
    ib_info.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    ib_info.size        = indices.size() * sizeof(uint32_t);       // 36 × 4 = 144 байта
    ib_info.usage       = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    ib_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationInfo ib_ai{};
    if (vmaCreateBuffer(ctx.allocator, &ib_info, &alloc_info,
                        &index_buffer, &index_buffer_alloc, &ib_ai) != VK_SUCCESS) {
        std::cerr << "[app] vmaCreateBuffer(index) failed\n";
        return false;
    }
    std::memcpy(ib_ai.pMappedData, indices.data(), ib_info.size);

    // Uniform-буфер (mat4 + vec4 = 80 байт)
    // 64 байта на матрицу 4×4 + 16 байт на vec4 цвета
    VkBufferCreateInfo ub_info{};
    ub_info.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    ub_info.size        = 64 + 16;
    ub_info.usage       = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    ub_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationInfo ub_ai{};
    if (vmaCreateBuffer(ctx.allocator, &ub_info, &alloc_info,
                        &uniform_buffer, &uniform_buffer_alloc, &ub_ai) != VK_SUCCESS) {
        std::cerr << "[app] vmaCreateBuffer(uniform) failed\n";
        return false;
    }
    uniform_buffer_mapped = ub_ai.pMappedData;

    //Descriptor Set Layout
    //один uniform-буфер в binding=0, используется в вершинном шейдере
    VkDescriptorSetLayoutBinding binding{};
    binding.binding         = 0;
    binding.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    binding.descriptorCount = 1;
    binding.stageFlags      = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo dsl_info{};
    dsl_info.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsl_info.bindingCount = 1;
    dsl_info.pBindings    = &binding;

    if (vkCreateDescriptorSetLayout(device, &dsl_info, nullptr, &descriptor_set_layout) != VK_SUCCESS) {
        std::cerr << "[app] vkCreateDescriptorSetLayout failed\n";
        return false;
    }

    //Descriptor Pool
    //из него будут выделяться DescriptorSets
    VkDescriptorPoolSize pool_size{};
    pool_size.type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    pool_size.descriptorCount = 1;

    VkDescriptorPoolCreateInfo dp_info{};
    dp_info.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dp_info.poolSizeCount = 1;
    dp_info.pPoolSizes    = &pool_size;
    dp_info.maxSets       = 1;

    if (vkCreateDescriptorPool(device, &dp_info, nullptr, &descriptor_pool) != VK_SUCCESS) {
        std::cerr << "[app] vkCreateDescriptorPool failed\n";
        return false;
    }

    // Descriptor Set
    // Выделяем один набор из Pool
    VkDescriptorSetAllocateInfo ds_alloc{};
    ds_alloc.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ds_alloc.descriptorPool     = descriptor_pool;
    ds_alloc.descriptorSetCount = 1;
    ds_alloc.pSetLayouts        = &descriptor_set_layout;

    if (vkAllocateDescriptorSets(device, &ds_alloc, &descriptor_set) != VK_SUCCESS) {
        std::cerr << "[app] vkAllocateDescriptorSets failed\n";
        return false;
    }

    // Связываем наш uniform_buffer с этим Set-ом
    VkDescriptorBufferInfo buf_info{};
    buf_info.buffer = uniform_buffer;
    buf_info.offset = 0;
    buf_info.range  = ub_info.size;

    VkWriteDescriptorSet write{};
    write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet          = descriptor_set;
    write.dstBinding      = 0;
    write.descriptorCount = 1;
    write.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    write.pBufferInfo     = &buf_info;

    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);   // Применяем изменения

    // Pipeline Layout
    // Говорит пайплайну - буду использовать этот descriptor set layout
    VkPipelineLayoutCreateInfo pl_info{};
    pl_info.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl_info.setLayoutCount = 1;
    pl_info.pSetLayouts    = &descriptor_set_layout;

    if (vkCreatePipelineLayout(device, &pl_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
        std::cerr << "[app] vkCreatePipelineLayout failed\n";
        return false;
    }

    // Shader Modules
    // Путь "../shaders/..." — потому что .exe лежит в build-debug, а шейдеры — в shaders/
    auto vert_code = readFile("../shaders/shader.vert.spv");
    auto frag_code = readFile("../shaders/shader.frag.spv");
    if (vert_code.empty() || frag_code.empty()) {
        std::cerr << "[app] Проверь путь до .spv (../shaders/)\n";
        return false;
    }

    VkShaderModule vert_module = createShaderModule(vert_code);
    VkShaderModule frag_module = createShaderModule(frag_code);
    if (!vert_module || !frag_module) return false;

    // Описываем две стадии пайплайна — vertex и fragment
    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert_module;
    stages[0].pName  = "main";           // имя точки входа в шейдере

    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag_module;
    stages[1].pName  = "main";

    // Vertex Input
    // Описываем формат данных в vertex_buffer: 12 байт на вершину, поле position
    VkVertexInputBindingDescription vertex_binding{};
    vertex_binding.binding   = 0;
    vertex_binding.stride    = sizeof(Vertex);
    vertex_binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription vertex_attr{};
    vertex_attr.location = 0;                        // В шейдере: layout(location = 0)
    vertex_attr.binding  = 0;
    vertex_attr.format   = VK_FORMAT_R32G32B32_SFLOAT;  // 3 × float
    vertex_attr.offset   = offsetof(Vertex, position);   // 0

    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi.vertexBindingDescriptionCount   = 1;
    vi.pVertexBindingDescriptions      = &vertex_binding;
    vi.vertexAttributeDescriptionCount = 1;
    vi.pVertexAttributeDescriptions    = &vertex_attr;

    // IA / VP / RS / MS / DS / CB
    // InputAssembly: собирает вершины
    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;      // каждый 3 индекса = треугольник

    // Viewport/Scissor — зададим позже в render (динамические)
    VkPipelineViewportStateCreateInfo vp{};
    vp.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1;
    vp.scissorCount  = 1;

    // Rasterization —как заливать треугольники
    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL;                    // залитые треугольники
    rs.cullMode    = VK_CULL_MODE_NONE;                       // не отсекаем задние грани (для наглядности)
    rs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth   = 1.0f;

    // Multisample — пока без сглаживания
    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // Depth/Stencil —Без этого задние грани рисовались бы поверх передних
    VkPipelineDepthStencilStateCreateInfo ds{};
    ds.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable  = VK_TRUE;                            // сравнивать глубину
    ds.depthWriteEnable = VK_TRUE;                            // записывать глубину
    ds.depthCompareOp   = VK_COMPARE_OP_LESS;                 // меньшая глубина = ближе

    // Color Blend — как смешивать новый цвет с существующим в буфере
    VkPipelineColorBlendAttachmentState cba{};
    cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                       | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    cba.blendEnable    = VK_FALSE;                            // без прозрачности

    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1;
    cb.pAttachments    = &cba;

    // Динамические states — viewport и scissor меняются во время выполнения (при ресайзе окна)
    VkDynamicState dyn_states[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dyn{};
    dyn.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn.dynamicStateCount = 2;
    dyn.pDynamicStates    = dyn_states;

    // Graphics Pipeline
    // Финальная сборка всего в единый объект
    VkGraphicsPipelineCreateInfo pipe_info{};
    pipe_info.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipe_info.stageCount          = 2;
    pipe_info.pStages             = stages;
    pipe_info.pVertexInputState   = &vi;
    pipe_info.pInputAssemblyState = &ia;
    pipe_info.pViewportState      = &vp;
    pipe_info.pRasterizationState = &rs;
    pipe_info.pMultisampleState   = &ms;
    pipe_info.pDepthStencilState  = &ds;
    pipe_info.pColorBlendState    = &cb;
    pipe_info.pDynamicState       = &dyn;
    pipe_info.layout              = pipeline_layout;
    pipe_info.renderPass          = ctx.render_pass;
    pipe_info.subpass             = 0;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipe_info,
                                  nullptr, &pipeline) != VK_SUCCESS) {
        std::cerr << "[app] vkCreateGraphicsPipelines failed\n";
        return false;
    }

    // Шейдерные модули больше не нужны — пайплайн их "вобрал" в себя
    vkDestroyShaderModule(device, vert_module, nullptr);
    vkDestroyShaderModule(device, frag_module, nullptr);

    return true;
}

// shutdown — уничтожаем всё в обратном порядке создания
void shutdown() {
    auto& ctx = graphics::internal::context;
    vkDeviceWaitIdle(ctx.device);    //пока GPU закончит все операции

    vkDestroyPipeline(ctx.device, pipeline, nullptr);
    vkDestroyPipelineLayout(ctx.device, pipeline_layout, nullptr);
    vkDestroyDescriptorPool(ctx.device, descriptor_pool, nullptr);
    // Set уничтожится автоматически вместе с Pool-ом
    vkDestroyDescriptorSetLayout(ctx.device, descriptor_set_layout, nullptr);

    // VMA сам уничтожает и буфер, и его аллокацию
    vmaDestroyBuffer(ctx.allocator, uniform_buffer, uniform_buffer_alloc);
    vmaDestroyBuffer(ctx.allocator, index_buffer,   index_buffer_alloc);
    vmaDestroyBuffer(ctx.allocator, vertex_buffer,  vertex_buffer_alloc);
}

// UPDATE — GUI + математика + запись данных в uniform
void update(double time_delta) {
    auto& ctx = graphics::internal::context;

    // ---------- GUI ----------
    ImGui::Begin("Scene Settings");                       // Открываем окно настроек
    ImGui::Checkbox("Perspective Projection", &is_perspective);
    ImGui::ColorEdit4("Object Color", color);             // 4 компоненты RGBA

    ImGui::Separator();                                   // Горизонтальная линия-разделитель
    ImGui::DragFloat3("Position", pos,   0.05f);          // 3 float с шагом 0.05
    ImGui::DragFloat3("Rotation", rot,   1.0f);           // Шаг 1°
    ImGui::DragFloat3("Scale",    scale, 0.05f);

    ImGui::Separator();
    ImGui::Checkbox("Play Animation", &is_playing);
    ImGui::SliderFloat("Speed",  &anim_speed,  0.1f, 5.0f);
    ImGui::SliderFloat("Radius", &anim_radius, 0.5f, 10.0f);
    ImGui::End();

    //Логика анимаци
    // Если включена — куб движется у: x = R·sin(t), z = R·sin(2t)
    if (is_playing) {
        current_time += static_cast<float>(time_delta * anim_speed);
        pos[0] = anim_radius * std::sin(current_time);
        pos[2] = anim_radius * std::sin(current_time * 2.0f);
    }

    //Матрица Model: T * Ry * Rx * Rz * S
    // Порядок важен: сначала масштаб, потом повороты, потом перенос.
    // Читаем справа налево: сначала умножаем на S, потом Rz, потом Rx, потом Ry, потом T.
    const float DEG2RAD = 3.14159265358979323846f / 180.0f;    // Коэффициент градусы - радианы
    Mat4 model = multiply(
        translate(pos[0], pos[1], pos[2]),
        multiply(rotY(rot[1] * DEG2RAD),
        multiply(rotX(rot[0] * DEG2RAD),
        multiply(rotZ(rot[2] * DEG2RAD),
                 scaleM(scale[0], scale[1], scale[2])))));

    // View
    // Камера в точке (0,0,5), смотрит в начало координат, верх — (0,1,0)
    float eye[3]    = {0.0f, 0.0f, 5.0f};
    float center[3] = {0.0f, 0.0f, 0.0f};
    float up[3]     = {0.0f, 1.0f, 0.0f};
    Mat4 view = lookAt(eye, center, up);

    //Projection
    float aspect = static_cast<float>(ctx.swapchain_extent.width)
                 / static_cast<float>(ctx.swapchain_extent.height);   // 1280/720 ≈ 1.78
    Mat4 proj;
    if (is_perspective) {
        proj = perspective(60.0f * DEG2RAD, aspect, 0.1f, 100.0f);    // FOV 60°, near 0.1, far 100
    } else {
        float s = 2.0f;
        proj = ortho(-s * aspect, s * aspect, -s, s, 0.1f, 100.0f);
    }

    // MVP
    // Порядок: proj × view × model. Слева направо — как матрицы применяются к точке.
    Mat4 mvp = multiply(proj, multiply(view, model));

    // Запись в uniform
    // Структура точно повторяет layout(std140) из GLSL: mat4 (64 байта) + vec4 (16 байт)
    struct UniformData {
        float mvp[16];
        float color[4];
    } data;
    std::memcpy(data.mvp,   mvp.m, sizeof(mvp.m));
    std::memcpy(data.color, color, sizeof(color));
    std::memcpy(uniform_buffer_mapped, &data, sizeof(data));   // В память GPU
}

// RENDER — команды рисования
void render(const graphics::internal::FrameData& fd) {
    auto& ctx = graphics::internal::context;

    // Начинаем запись команд в буфер, который передал graphics_internal::prepare()
    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;   // буфер используем один раз
    vkBeginCommandBuffer(fd.command_buffer, &begin_info);

    //Значения очистки: цвет фона и глубина (по умолчанию 1.0 = "далеко")
    VkClearValue clear_values[2] = {};
    clear_values[0].color        = {{0.1f, 0.1f, 0.1f, 1.0f}};    // тёмно-серый фон
    clear_values[1].depthStencil = {1.0f, 0};

    //Начинаем render pass с нашим framebuffer
    VkRenderPassBeginInfo rp{};
    rp.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rp.renderPass        = ctx.render_pass;
    rp.framebuffer       = fd.framebuffer;
    rp.renderArea.offset = {0, 0};
    rp.renderArea.extent = ctx.swapchain_extent;
    rp.clearValueCount   = 2;
    rp.pClearValues      = clear_values;

    vkCmdBeginRenderPass(fd.command_buffer, &rp, VK_SUBPASS_CONTENTS_INLINE);

    // Привязываем пайплайн (все состояния + шейдеры)
    vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

    // Viewport — прямоугольник на экране, в который рендерим (весь экран)
    VkViewport viewport{};
    viewport.x        = 0.0f;
    viewport.y        = 0.0f;
    viewport.width    = static_cast<float>(ctx.swapchain_extent.width);
    viewport.height   = static_cast<float>(ctx.swapchain_extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);

    // Scissor — область обрезки (весь экран)
    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = ctx.swapchain_extent;
    vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

    // Привязываем буферы вершин и индексов
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vertex_buffer, offsets);
    vkCmdBindIndexBuffer(fd.command_buffer, index_buffer, 0, VK_INDEX_TYPE_UINT32);

    // Привязываем descriptor set (наш uniform-буфер)
    vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);

    // Рисование: 36 индексов, 1 экземпляр, смещения = 0
    vkCmdDrawIndexed(fd.command_buffer, static_cast<uint32_t>(indices.size()),
                     1, 0, 0, 0);

    vkCmdEndRenderPass(fd.command_buffer);       // Закрываем render pass
    vkEndCommandBuffer(fd.command_buffer);       // Заканчиваем запись команд
}

} // namespace application