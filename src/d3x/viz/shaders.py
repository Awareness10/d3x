"""GLSL shaders for d3x visualization."""

# Body rendering - instanced spheres approximated as screen-space circles
BODY_VERTEX = """
#version 330 core

uniform mat4 view_proj;
uniform float viewport_height;

in vec3 in_position;   // Instance position
in vec3 in_color;      // Instance color
in float in_size;      // Instance size (radius)

out vec3 v_color;

void main() {
    v_color = in_color;
    gl_Position = view_proj * vec4(in_position, 1.0);

    // Point size based on size relative to clip depth (stable at all angles)
    float dist = gl_Position.w;  // Distance from camera
    float proj_size = (in_size / dist) * viewport_height * 0.5;
    gl_PointSize = clamp(proj_size, 6.0, 200.0);
}
"""

BODY_FRAGMENT = """
#version 330 core

in vec3 v_color;

out vec4 frag_color;

void main() {
    // Circular point with soft edge
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    float dist = length(coord);

    if (dist > 1.0) discard;

    // Soft edge
    float alpha = 1.0 - smoothstep(0.7, 1.0, dist);

    // Simple shading - darker at edges
    float shade = 1.0 - dist * 0.3;

    frag_color = vec4(v_color * shade, alpha);
}
"""

# Trail rendering - line strips with alpha fade
TRAIL_VERTEX = """
#version 330 core

uniform mat4 view_proj;

in vec3 in_position;
in float in_age;       // 0.0 = newest, 1.0 = oldest
in vec3 in_color;

out float v_age;
out vec3 v_color;

void main() {
    v_age = in_age;
    v_color = in_color;
    gl_Position = view_proj * vec4(in_position, 1.0);
}
"""

TRAIL_FRAGMENT = """
#version 330 core

in float v_age;
in vec3 v_color;

out vec4 frag_color;

void main() {
    // Fade alpha with age, but keep minimum visibility
    float alpha = 1.0 - v_age * 0.92;
    frag_color = vec4(v_color, alpha);
}
"""

# Grid rendering - simple lines (no fade for now)
GRID_VERTEX = """
#version 330 core

uniform mat4 view_proj;

in vec3 in_position;

void main() {
    gl_Position = view_proj * vec4(in_position, 1.0);
}
"""

GRID_FRAGMENT = """
#version 330 core

uniform vec4 grid_color;

out vec4 frag_color;

void main() {
    frag_color = grid_color;
}
"""

# Axis indicator - RGB = XYZ (world-space, toggleable)
AXIS_VERTEX = """
#version 330 core

uniform mat4 view_proj;

in vec3 in_position;
in vec3 in_color;

out vec3 v_color;

void main() {
    v_color = in_color;
    gl_Position = view_proj * vec4(in_position, 1.0);
}
"""

AXIS_FRAGMENT = """
#version 330 core

in vec3 v_color;

out vec4 frag_color;

void main() {
    frag_color = vec4(v_color, 1.0);
}
"""

# Corner gizmo - orientation indicator fixed to screen corner
GIZMO_VERTEX = """
#version 330 core

uniform mat4 rotation;      // Camera rotation only (no translation)
uniform vec2 offset;        // Screen position offset (NDC)
uniform float scale;        // Size scale

in vec3 in_position;
in vec3 in_color;

out vec3 v_color;

void main() {
    v_color = in_color;
    // Rotate axis by camera orientation
    vec4 rotated = rotation * vec4(in_position * scale, 1.0);
    // Position in corner (NDC space)
    gl_Position = vec4(rotated.xy + offset, 0.0, 1.0);
}
"""

GIZMO_FRAGMENT = """
#version 330 core

in vec3 v_color;

out vec4 frag_color;

void main() {
    frag_color = vec4(v_color, 1.0);
}
"""

FPS_VERTEX = """
#version 330
in vec2 in_vert;
uniform vec2 offset;
uniform vec2 scale;
void main() {
    gl_Position = vec4(in_vert * scale + offset, 0.0, 1.0);
}
"""

FPS_FRAGMENT = """
#version 330
out vec4 f_color;
uniform vec3 color;
void main() {
    f_color = vec4(color, 1.0);
}
"""

TEXT_VERTEX = """
#version 330
in vec2 in_vert;      // Basic quad [-1, 1]
in vec2 in_pos;       // Screen pixel position (e.g., 1200, 700)
in int in_digit;      // 0-9

uniform vec2 screen_size;
out vec2 v_uv;

void main() {
    // 1. Calculate UV: Select the 1/10th slice of the atlas
    float u_width = 0.1; 
    float u_start = float(in_digit) * u_width;
    v_uv = vec2(u_start + (in_vert.x * 0.5 + 0.5) * u_width, 1.0 - (in_vert.y * 0.5 + 0.5));
    
    // 2. Scale the quad size (e.g., 24x24 pixels)
    vec2 char_size = vec2(24.0, 24.0);
    vec2 pixel_pos = in_pos + (in_vert * 0.5 + 0.5) * char_size;
    
    // 3. Convert Pixels to NDC: (pos / screen * 2.0) - 1.0
    // We flip Y because screen coords are top-down, NDC is bottom-up
    vec2 ndc = (pixel_pos / screen_size) * 2.0 - 1.0;
    gl_Position = vec4(ndc.x, ndc.y, 0.0, 1.0);
}
"""

TEXT_FRAGMENT = """
#version 330
uniform sampler2D u_atlas;
out vec4 f_color;
in vec2 v_uv;

void main() {
    f_color = texture(u_atlas, v_uv);
    if (f_color.a < 0.1) discard; // Transparency
}
"""
