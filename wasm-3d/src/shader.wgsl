// Vertex shader

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
    @location(2) normal: vec3<f32>,
}

struct InstanceInput {
    @location(5) model_matrix_0: vec4<f32>,
    @location(6) model_matrix_1: vec4<f32>,
    @location(7) model_matrix_2: vec4<f32>,
    @location(8) model_matrix_3: vec4<f32>,
    @location(9) normal_matrix_0: vec3<f32>,
    @location(10) normal_matrix_1: vec3<f32>,
    @location(11) normal_matrix_2: vec3<f32>
};


struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) world_position: vec3<f32>,
};

// Vertex shader
struct CameraUniform {
    view_proj: mat4x4<f32>,
    position: vec4<f32>,
};
@group(1) @binding(0) // 1.
var<uniform> camera: CameraUniform;

struct LightUniform {
    position: vec3<f32>,
    colour: vec3<f32>,
};
@group(2) @binding(0)
var<uniform> light: LightUniform;


@vertex
fn vs_main(
    vert: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    let model_matrix = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );
    let normal_matrix = mat3x3<f32>(
        instance.normal_matrix_0,
        instance.normal_matrix_1,
        instance.normal_matrix_2
    );

    var out: VertexOutput;
    out.tex_coords = vert.tex_coords;
    out.world_normal = normal_matrix * vert.normal;
    let world_position: vec4<f32> = model_matrix * vec4<f32>(vert.position, 1.0);
    out.world_position = world_position.xyz;
    out.clip_position = camera.view_proj * world_position;
    return out;
}


@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let ambient_strength = 0.1;
    let texture_colour: vec4<f32> = textureSample(t_diffuse, s_diffuse, in.tex_coords);
    let ambient_light_colour = ambient_strength * light.colour;

    let light_dir = normalize(light.position - in.world_position);
    let brightness = max(dot(normalize(in.world_normal), light_dir), 0.0);
    let diffuse_colour = brightness * light.colour;

    let surface_to_light = normalize(light.position - in.world_position);
    let camera_reflection = normalize(reflect(in.world_position - camera.position.xyz, in.world_normal));

    let specular = pow(max(0.0, dot(surface_to_light, camera_reflection)), 32.0);
    let specular_colour = light.colour * specular;


    let result = texture_colour.xyz * (ambient_light_colour + diffuse_colour + specular_colour);
    return vec4<f32>(result, 1.0);
 }
