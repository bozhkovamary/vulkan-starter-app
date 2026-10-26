#version 450 core

// Входные атрибуты вершины 
layout(location = 0) in vec3 in_position;   // позиция в локальных координатах
layout(location = 1) in vec3 in_normal;     // нормаль (сгенерирована процедурно)

// Выходные данные для фрагментного шейдера
layout(location = 0) out vec3 out_world_pos;   // позиция в мировых координатах
layout(location = 1) out vec3 out_normal;      // нормаль в мировых координатах

//SET 0: общие данные сцены
layout(std140, set = 0, binding = 0) uniform SceneUniforms {
    mat4 view_projection;               // P * V — из C++ уже перемножено
    vec4 camera_position;               // xyz — позиция камеры
    vec4 directional_light_direction;   // xyz — направление источника
    vec4 directional_light_color;       // rgb — цвет источника
    vec4 ambient_light;                 // rgb — цвет ambient
    uvec4 light_counts;                 // x = кол-во point, y = кол-во spot
} scene;

// ===== SET 1: данные конкретной модели =====
layout(std140, set = 1, binding = 0) uniform ModelUniform {
    mat4 model;         // модельная матрица
    vec4 albedo;        // rgb — базовый цвет
    vec4 specular;      // rgb — цвет блика, a = shininess (коэффициент блеска)
} model;

// ===== PUSH CONSTANTS: для визуализации источников =====
// Если flags & 1 == 1 — рисуем визуализацию, используем model_override и color_override,
// игнорируем model.model и не применяем освещение.
layout(push_constant) uniform Push {
    mat4 model_override;   // модельная матрица для визуализации источника
    vec4 color_override;   // цвет визуализации
    uint flags;            // бит 0 = unlit
} pc;

void main() {
    // Выбираем модельную матрицу: override, если это визуализация
    bool unlit = (pc.flags & 1u) != 0u;
    mat4 used_model = unlit ? pc.model_override : model.model;

    // Мировая позиция вершины
    vec4 world = used_model * vec4(in_position, 1.0);
    gl_Position = scene.view_projection * world;

    // Передаём в фрагментный шейдер мировую позицию и нормаль
    out_world_pos = world.xyz;
    // Нормаль трансформируется только поворотом, но пока модель не имеет
    // неравномерного масштаба — можно использовать mat3 модельной матрицы.
    out_normal = normalize(mat3(used_model) * in_normal);
}