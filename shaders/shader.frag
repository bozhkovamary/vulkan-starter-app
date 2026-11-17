#version 450 core

layout(location = 0) in vec3 in_world_pos;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;

layout(location = 0) out vec4 out_color;

struct PointLight {
    vec4 position;
    vec4 color;
};

struct SpotLight {
    vec4 position;
    vec4 direction;
    vec4 color;
    vec4 angles; // x = внутренний угол, y = внешний угол
};

// SET 0: сцена и источники света
layout(std140, set = 0, binding = 0) uniform SceneUniforms {
    mat4 view_projection;
    vec4 camera_position;
    vec4 directional_light_direction;
    vec4 directional_light_color;
    vec4 ambient_light;
    uvec4 light_counts;
} scene;

layout(std430, set = 0, binding = 1) readonly buffer PointLights {
    PointLight point_lights[];
};

layout(std430, set = 0, binding = 2) readonly buffer SpotLights {
    SpotLight spot_lights[];
};

// SET 1: динамический UBO модели + 3 текстуры материала
layout(std140, set = 1, binding = 0) uniform ModelUniforms {
    mat4 model;
    vec4 albedo;
    vec4 specular; // rgb + shininess в .a
    uvec4 flags;   // x = unlit
} model_ubo;

layout(set = 1, binding = 1) uniform sampler2D albedo_sampler;
layout(set = 1, binding = 2) uniform sampler2D specular_sampler;
layout(set = 1, binding = 3) uniform sampler2D emissive_sampler;

// PUSH CONSTANTS: управление нетривиальным сэмплированием
layout(std140, push_constant) uniform PipelineConsts {
    uint nontrivial_sampling;
    float time;
    float emissive_strength;
    float wave_strength;
} pc;

vec3 blinn_phong(vec3 N, vec3 L, vec3 V, vec3 light_color, float attenuation,
                 vec3 albedo, vec3 spec_color, float shininess) {
    float NdotL = max(dot(N, L), 0.0);
    vec3 diffuse = albedo * NdotL;

    vec3 H = normalize(L + V);
    float NdotH = max(dot(N, H), 0.0);
    vec3 specular = spec_color * pow(NdotH, shininess);

    return (diffuse + specular) * light_color * attenuation;
}

void main() {
    // Если рисуется визуализация источника света (unlit)
    if (model_ubo.flags.x != 0u) {
        out_color = vec4(model_ubo.albedo.rgb, 1.0);
        return;
    }

    vec2 uv = in_uv;
    vec3 sampled_albedo;
    vec3 sampled_specular;
    vec3 sampled_emissive;

    // Доп. задание 2: Нетривиальное сэмплирование (управляется через push-константы)
    if (pc.nontrivial_sampling != 0u) {
        // 1. Набор UV №1: волновая деформация во времени
        vec2 wave = vec2(
            sin(uv.y * 10.0 + pc.time * 2.5),
            cos(uv.x * 10.0 + pc.time * 2.0)
        ) * pc.wave_strength;
        vec2 uv1 = uv + wave;

        // 2. Набор UV №2: масштабирование и противофазное движение для множественного сэмплирования
        float angle = pc.time * 0.4;
        mat2 rot = mat2(cos(angle), -sin(angle), sin(angle), cos(angle));
        vec2 uv2 = rot * (uv - 0.5) * 1.5 + 0.5 + vec2(pc.time * 0.1, -pc.time * 0.08);

        // Множественное сэмплирование одного и того же изображения разными UV
        vec3 tex1 = texture(albedo_sampler, uv1).rgb;
        vec3 tex2 = texture(albedo_sampler, uv2).rgb;
        vec3 combined_tex = mix(tex1, tex2, 0.35);

        // Модуляция цвета сэмплированного изображения от времени
        vec3 color_modulation = 0.75 + 0.25 * cos(pc.time * 2.0 + vec3(0.0, 2.1, 4.2) + uv.xyx * 3.14159);
        sampled_albedo = combined_tex * color_modulation;

        sampled_specular = texture(specular_sampler, uv1).rgb;
        sampled_emissive = texture(emissive_sampler, uv1).rgb;
    } else {
        // Обычное сэмплирование
        sampled_albedo   = texture(albedo_sampler, uv).rgb;
        sampled_specular = texture(specular_sampler, uv).rgb;
        sampled_emissive = texture(emissive_sampler, uv).rgb;
    }

    vec3 N = normalize(in_normal);
    vec3 V = normalize(scene.camera_position.xyz - in_world_pos);

    vec3 albedo = sampled_albedo * model_ubo.albedo.rgb;
    // Доп. задание 3: Specular-текстура определяет, какая часть геометрии блестит
    vec3 spec   = sampled_specular * model_ubo.specular.rgb;
    float shin  = model_ubo.specular.a;

    // Ambient
    vec3 color = scene.ambient_light.rgb * albedo;

    // 1. Направленный источник
    {
        vec3 L = normalize(-scene.directional_light_direction.xyz);
        color += blinn_phong(N, L, V, scene.directional_light_color.rgb, 1.0, albedo, spec, shin);
    }

    // 2. Точечные источники
    uint np = scene.light_counts.x;
    for (uint i = 0u; i < np; ++i) {
        vec3 to_light = point_lights[i].position.xyz - in_world_pos;
        float dist = max(length(to_light), 0.0001);
        vec3 L = to_light / dist;
        float atten = 1.0 / (dist * dist);

        color += blinn_phong(N, L, V, point_lights[i].color.rgb, atten, albedo, spec, shin);
    }

    // 3. Прожекторные источники
    uint ns = scene.light_counts.y;
    for (uint i = 0u; i < ns; ++i) {
        vec3 to_light = spot_lights[i].position.xyz - in_world_pos;
        float dist = max(length(to_light), 0.0001);
        vec3 L = to_light / dist;

        vec3 axis = normalize(spot_lights[i].direction.xyz);
        float cosA = dot(-L, axis);
        float cosOuter = cos(radians(spot_lights[i].angles.y));
        float cosInner = cos(radians(spot_lights[i].angles.x));

        float spot = clamp((cosA - cosOuter) / max(cosInner - cosOuter, 0.0001), 0.0, 1.0);
        float atten = spot / (dist * dist);

        color += blinn_phong(N, L, V, spot_lights[i].color.rgb, atten, albedo, spec, shin);
    }

    // Доп. задание 3: Emissive-текстура суммируется после расчета Блинн-Фонга (светится в темноте)
    color += sampled_emissive * pc.emissive_strength;

    out_color = vec4(color, 1.0);
}