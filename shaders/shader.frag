#version 450 core

layout(location = 0) in vec3 in_world_pos;
layout(location = 1) in vec3 in_normal;

layout(location = 0) out vec4 out_color;

// ===== Структуры источников =====
struct PointLight {
    vec4 position;   // xyz — позиция
    vec4 color;      // rgb — цвет
};

struct SpotLight {
    vec4 position;   // xyz — позиция
    vec4 direction;  // xyz — направление
    vec4 color;      // rgb — цвет
    vec4 angles;     // x = внутренний угол (град), y = внешний угол (град)
};

// ===== Сцена (set 0) =====
layout(std140, set = 0, binding = 0) uniform SceneUniforms {
    mat4 view_projection;
    vec4 camera_position;
    vec4 directional_light_direction;
    vec4 directional_light_color;
    vec4 ambient_light;
    uvec4 light_counts;
} scene;

// ===== Storage буферы с массивами источников =====
// std430 — для storage буферов (плотная упаковка)
layout(std430, set = 0, binding = 1) readonly buffer PointLights {
    PointLight point_lights[];
};

layout(std430, set = 0, binding = 2) readonly buffer SpotLights {
    SpotLight spot_lights[];
};

// ===== Модель (set 1) =====
layout(std140, set = 1, binding = 0) uniform ModelUniform {
    mat4 model;
    vec4 albedo;
    vec4 specular;   // rgb + shininess в .a
} model;

// ===== Push constants =====
layout(push_constant) uniform Push {
    mat4 model_override;
    vec4 color_override;
    uint flags;
} pc;

// ===== Блинн-Фонг =====
vec3 blinn_phong(vec3 N, vec3 L, vec3 V, vec3 light_color, float attenuation,
                 vec3 albedo, vec3 spec_color, float shininess) {
    // Диффузная часть: Ламберт
    float NdotL = max(dot(N, L), 0.0);
    vec3 diffuse = albedo * NdotL;

    // Блик: половинный вектор
    vec3 H = normalize(L + V);
    float NdotH = max(dot(N, H), 0.0);
    vec3 specular = spec_color * pow(NdotH, shininess);

    return (diffuse + specular) * light_color * attenuation;
}

void main() {
    // Визуализация источника — выводим цвет как есть, без освещения
    if ((pc.flags & 1u) != 0u) {
        out_color = pc.color_override;
        return;
    }

    vec3 N = normalize(in_normal);
    vec3 V = normalize(scene.camera_position.xyz - in_world_pos);

    vec3 albedo   = model.albedo.rgb;
    vec3 spec     = model.specular.rgb;
    float shin    = model.specular.a;

    // Ambient
    vec3 color = scene.ambient_light.rgb * albedo;

    // --- Направленный источник ---
    {
        vec3 L = normalize(-scene.directional_light_direction.xyz);
        color += blinn_phong(N, L, V, scene.directional_light_color.rgb,
                             1.0, albedo, spec, shin);
    }

    // --- Точечные источники (storage buffer) ---
    uint np = scene.light_counts.x;
    for (uint i = 0u; i < np; ++i) {
        vec3 to_light = point_lights[i].position.xyz - in_world_pos;
        float dist = max(length(to_light), 0.0001);
        vec3 L = to_light / dist;

        // Закон обратных квадратов
        float atten = 1.0 / (dist * dist);

        color += blinn_phong(N, L, V, point_lights[i].color.rgb,
                             atten, albedo, spec, shin);
    }

    // --- Прожекторные источники (storage buffer) ---
    uint ns = scene.light_counts.y;
    for (uint i = 0u; i < ns; ++i) {
        vec3 to_light = spot_lights[i].position.xyz - in_world_pos;
        float dist = max(length(to_light), 0.0001);
        vec3 L = to_light / dist;

        vec3 axis = normalize(spot_lights[i].direction.xyz);
        // cos угла между направлением на фрагмент и осью прожектора
        float cosA = dot(-L, axis);

        float cosOuter = cos(radians(spot_lights[i].angles.y));
        float cosInner = cos(radians(spot_lights[i].angles.x));

        // Мягкий край: плавный переход от inner к outer
        float spot = clamp((cosA - cosOuter) / max(cosInner - cosOuter, 0.0001),
                           0.0, 1.0);

        float atten = spot / (dist * dist);

        color += blinn_phong(N, L, V, spot_lights[i].color.rgb,
                             atten, albedo, spec, shin);
    }

    out_color = vec4(color, 1.0);
}