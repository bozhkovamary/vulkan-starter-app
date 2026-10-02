#version 450 core

// Входные атрибуты из вершинного буфера
layout(location = 0) in vec3 in_position;

// Uniform-буфер с выравниванием std140
layout(std140, set = 0, binding = 0) uniform GlobalUniforms {
    mat4 mvp;         // Матрица Model-View-Projection (64 байта)
    vec4 base_color;  // Базовый цвет из ImGUI (16 байт)
} ubo;

// Выходной цвет для фрагментного шейдера
layout(location = 0) out vec4 out_color;

void main() {
    gl_Position = ubo.mvp * vec4(in_position, 1.0);
    
    // Задание 5: Процедурный цвет на основе локальной позиции.
    // Сдвигаем координаты от [-1, 1] к [0, 1] для корректного отображения RGB
    vec3 local_color = in_position * 0.5 + 0.5;
    
    // Итоговый цвет = локальный цвет вершины * цвет из интерфейса
    out_color = vec4(local_color, 1.0) * ubo.base_color;
}