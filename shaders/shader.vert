#version 450 core

// Входные атрибуты вершины (в VRAM)
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;

// Выходные данные во фрагментный шейдер
layout(location = 0) out vec3 out_world_pos;
layout(location = 1) out vec3 out_normal;
layout(location = 2) out vec2 out_uv;

// SET 0: общие данные сцены
layout(std140, set = 0, binding = 0) uniform SceneUniforms {
    mat4 view_projection;
    vec4 camera_position;
    vec4 directional_light_direction;
    vec4 directional_light_color;
    vec4 ambient_light;
    uvec4 light_counts;
} scene;

// SET 1: динамический UBO модели (VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC)
layout(std140, set = 1, binding = 0) uniform ModelUniforms {
    mat4 model;
    vec4 albedo;      // rgb — базовый цвет (для unlit-визуализации или множителя)
    vec4 specular;    // rgb — цвет блика, a — shininess
    uvec4 flags;      // x = unlit (1 для визуализации источников)
} model_ubo;

// PUSH CONSTANTS: только управление нетривиальным сэмплированием и время
layout(std140, push_constant) uniform PipelineConsts {
    uint nontrivial_sampling;
    float time;
    float emissive_strength;
    float wave_strength;
} pc;

void main() {
    vec4 world = model_ubo.model * vec4(in_position, 1.0);
    gl_Position = scene.view_projection * world;

    out_world_pos = world.xyz;
    out_normal = normalize(mat3(model_ubo.model) * in_normal);
    out_uv = in_uv;
}