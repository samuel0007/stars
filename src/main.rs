use bytemuck::{Pod, Zeroable};
use nannou::prelude::*;
use nannou::wgpu::{self, util::DeviceExt};
use nannou::Frame;

const N_PARTICLES: usize = 100_000;
const GRID_SIZE: u32 = 64;
const MAX_PER_CELL: u32 = 64;
const NUM_CELLS: u32 = GRID_SIZE * GRID_SIZE;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Particle {
    pos: [f32; 2],
    vel: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params {
    mouse: [f32; 2], // NDC
    radius2: f32,
    strength: f32,
}

struct Model {
    // render
    render_pipeline: wgpu::RenderPipeline,

    // compute pipelines
    clear_counts_pipeline: wgpu::ComputePipeline,
    build_grid_pipeline: wgpu::ComputePipeline,
    interact_pipeline: wgpu::ComputePipeline,

    // particle ping-pong
    buf_a: wgpu::Buffer,
    buf_b: wgpu::Buffer,
    source_is_a: bool,

    // grid buffers
    grid_counts: wgpu::Buffer,
    grid_indices: wgpu::Buffer,

    // bind groups
    clear_counts_bg: wgpu::BindGroup,
    build_bg_a: wgpu::BindGroup,
    build_bg_b: wgpu::BindGroup,
    interact_bg_a2b: wgpu::BindGroup,
    interact_bg_b2a: wgpu::BindGroup,

    // mouse params
    params_buf: wgpu::Buffer,
    params_bg: wgpu::BindGroup,
}

fn main() {
    nannou::app(model).update(update).view(view).run();
}

fn model(app: &App) -> Model {
    let win_id = app
        .new_window()
        .size(1024, 1024)
        .title("GPU particles: grid neighbor search + mouse repel")
        .msaa_samples(1)
        .build()
        .unwrap();

    let window = app.window(win_id).unwrap();
    let device = window.device();

    // particles
    let mut particles = Vec::with_capacity(N_PARTICLES);
    for _ in 0..N_PARTICLES {
        particles.push(Particle {
            // pos: [random_range(-1.0, 1.0), random_range(-1.0, 1.0)],
            pos: [random_range(-0.01, 0.01), random_range(-0.01, 0.01)],
            vel: [random_range(-0.0001, 0.0001), random_range(-0.0001, 0.0001)],
        });
    }

    let buf_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("particles A"),
        contents: bytemuck::cast_slice(&particles),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE,
    });
    let buf_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("particles B"),
        contents: bytemuck::cast_slice(&particles),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE,
    });

    // grid buffers
    let zero_counts = vec![0u32; NUM_CELLS as usize];
    let zero_indices = vec![0u32; (NUM_CELLS * MAX_PER_CELL) as usize];
    let grid_counts = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("grid_counts"),
        contents: bytemuck::cast_slice(&zero_counts),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let grid_indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("grid_indices"),
        contents: bytemuck::cast_slice(&zero_indices),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    // mouse params buffer
    let initial_params = Params {
        mouse: [0.0, 0.0],
        radius2: 0.05,     // ~radius 0.223 in NDC
        strength: 0.0015,  // tweak
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::bytes_of(&initial_params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // WGSL
    let vs_src = r#"
struct VsOut {
  @builtin(position) pos : vec4<f32>,
  @location(0) vel : vec2<f32>,
};

@vertex
fn vs_main(@location(0) pos: vec2<f32>, @location(1) vel: vec2<f32>) -> VsOut {
  var out: VsOut;
  out.pos = vec4<f32>(pos, 0.0, 1.0);
  out.vel = vel;
  return out;
}
"#;

    let fs_src = r#"
@fragment
fn fs_main(@location(0) vel: vec2<f32>) -> @location(0) vec4<f32> {
    let speed = length(vel) * 300.0;   // scale for visibility
    let t = clamp(speed, 0.0, 1.0);

    // colormap: black → blue → cyan → green → yellow → red → white
    var r: f32;
    var g: f32;
    var b: f32;

    if (t < 0.2) {
        r = 0.0;
        g = 0.0;
        b = t / 0.2;
    } else if (t < 0.4) {
        r = 0.0;
        g = (t - 0.2) / 0.2;
        b = 1.0;
    } else if (t < 0.6) {
        r = (t - 0.4) / 0.2;
        g = 1.0;
        b = 1.0 - (t - 0.4) / 0.2;
    } else if (t < 0.8) {
        r = 1.0;
        g = 1.0 - (t - 0.6) / 0.2;
        b = 0.0;
    } else {
        r = 1.0;
        g = (t - 0.8) / 0.2;
        b = (t - 0.8) / 0.2;
    }

    return vec4<f32>(r, g, b, 0.8);
}
"#;

    let cs_clear_src = format!(
        r#"
const GRID_SIZE : u32 = {gs}u;
const NUM_CELLS : u32 = 4096u;

struct Counts {{ data: array<atomic<u32>> }};

@group(0) @binding(0) var<storage, read_write> grid_counts : Counts;

@compute @workgroup_size(256)
fn clear_main(@builtin(global_invocation_id) id : vec3<u32>) {{
  let i = id.x;
  if (i < NUM_CELLS) {{
    atomicStore(&grid_counts.data[i], 0u);
  }}
}}
"#,
        gs = GRID_SIZE
    );

    let cs_build_src = format!(
        r#"
const GRID_SIZE    : u32 = {gs}u;
const NUM_CELLS    : u32 = 4096u;
const MAX_PER_CELL : u32 = {mpc}u;

struct Particles {{ data: array<vec4<f32>> }};
struct Counts    {{ data: array<atomic<u32>> }};
struct Grid      {{ data: array<u32> }};

@group(0) @binding(0) var<storage, read>       particles   : Particles;
@group(0) @binding(1) var<storage, read_write> grid_counts : Counts;
@group(0) @binding(2) var<storage, read_write> grid_indices: Grid;

fn cell_index(pos : vec2<f32>) -> u32 {{
    let gx_max = GRID_SIZE - 1u;
    let gy_max = GRID_SIZE - 1u;
    let gx = clamp(u32(((pos.x + 1.0) * 0.5) * f32(GRID_SIZE)), 0u, gx_max);
    let gy = clamp(u32(((pos.y + 1.0) * 0.5) * f32(GRID_SIZE)), 0u, gy_max);
    return gy * GRID_SIZE + gx;
}}

@compute @workgroup_size(256)
fn build_main(@builtin(global_invocation_id) id : vec3<u32>) {{
    let i = id.x;
    if (i >= arrayLength(&particles.data)) {{ return; }}
    let pos = particles.data[i].xy;

    let ci = cell_index(pos);
    let offset = atomicAdd(&grid_counts.data[ci], 1u);
    if (offset < MAX_PER_CELL) {{
        let idx = ci * MAX_PER_CELL + offset;
        grid_indices.data[idx] = i;
    }}
}}
"#,
        gs = GRID_SIZE,
        mpc = MAX_PER_CELL
    );

    let cs_interact_src = format!(
        r#"
const GRID_SIZE    : u32 = {gs}u;
const NUM_CELLS    : u32 = 4096u;
const MAX_PER_CELL : u32 = {mpc}u;

const RADIUS2   : f32 = 0.003;
const STRENGTH  : f32 = 0.00001;
const DAMPING   : f32 = 0.99;
const DT        : f32 = 1.0;
const EPS       : f32 = 1e-6;

struct Particles {{ data: array<vec4<f32>> }};
struct Counts    {{ data: array<atomic<u32>> }};
struct Grid      {{ data: array<u32> }};

struct Params {{
  mouse    : vec2<f32>,
  radius2  : f32,
  strength : f32,
}};

@group(0) @binding(0) var<storage, read>       in_particles  : Particles;
@group(0) @binding(1) var<storage, read_write> out_particles : Particles;
@group(0) @binding(2) var<storage, read>       grid_counts   : Counts;
@group(0) @binding(3) var<storage, read>       grid_indices  : Grid;
@group(1) @binding(0) var<uniform>             params        : Params;

fn grid_coord(pos : vec2<f32>) -> vec2<u32> {{
    let gx_max = GRID_SIZE - 1u;
    let gy_max = GRID_SIZE - 1u;
    let gx = clamp(u32(((pos.x + 1.0) * 0.5) * f32(GRID_SIZE)), 0u, gx_max);
    let gy = clamp(u32(((pos.y + 1.0) * 0.5) * f32(GRID_SIZE)), 0u, gy_max);
    return vec2<u32>(gx, gy);
}}

fn cell_index_xy(x: u32, y: u32) -> u32 {{
    return y * GRID_SIZE + x;
}}

@compute @workgroup_size(256)
fn interact_main(@builtin(global_invocation_id) id : vec3<u32>) {{
    let i = id.x;
    if (i >= arrayLength(&in_particles.data)) {{ return; }}

    var p = in_particles.data[i];
    var pos = p.xy;
    var vel = p.zw;

    // let dm   = pos - params.mouse; // repulsion
    let dm  = params.mouse - pos;   // attraction

    let d2m  = dot(dm, dm);
    if (d2m < params.radius2) {{
        let inv = inverseSqrt(d2m + EPS);
        let dir = dm * inv;
        vel += params.strength * dir;
    }}

    let gc = grid_coord(pos);

    // neighbor repulsion in 3x3 cells
    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {{
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {{
            let nx = i32(gc.x) + dx;
            let ny = i32(gc.y) + dy;
            if (nx < 0 || ny < 0 || nx >= i32(GRID_SIZE) || ny >= i32(GRID_SIZE)) {{ continue; }}
            let ci = cell_index_xy(u32(nx), u32(ny));
            let count = atomicLoad(&grid_counts.data[ci]);
            for (var j: u32 = 0u; j < count && j < MAX_PER_CELL; j = j + 1u) {{
                let pid = grid_indices.data[ci * MAX_PER_CELL + j];
                if (pid == i) {{ continue; }}
                let q = in_particles.data[pid];
                let d = q.xy - pos;
                let r2 = dot(d, d);
                if (r2 > EPS && r2 < RADIUS2) {{
                    let inv_r = inverseSqrt(r2);
                    let dir = d * inv_r;
                    vel -= STRENGTH * dir;
                }}
            }}
        }}
    }}

    vel *= DAMPING;
    pos += vel * DT;

    if (pos.x > 1.0 || pos.x < -1.0) {{ vel.x = -vel.x; }}
    if (pos.y > 1.0 || pos.y < -1.0) {{ vel.y = -vel.y; }}

    out_particles.data[i] = vec4<f32>(pos, vel);
}}
"#,
        gs = GRID_SIZE,
        mpc = MAX_PER_CELL
    );

    // shader modules
    let vs = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("vs"),
        source: wgpu::ShaderSource::Wgsl(vs_src.into()),
    });
    let fs = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("fs"),
        source: wgpu::ShaderSource::Wgsl(fs_src.into()),
    });
    let cs_clear = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cs_clear"),
        source: wgpu::ShaderSource::Wgsl(cs_clear_src.into()),
    });
    let cs_build = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cs_build"),
        source: wgpu::ShaderSource::Wgsl(cs_build_src.into()),
    });
    let cs_interact = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cs_interact"),
        source: wgpu::ShaderSource::Wgsl(cs_interact_src.into()),
    });

    // render pipeline
    let render_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("render layout"),
        bind_group_layouts: &[],
        push_constant_ranges: &[],
    });
    let particle_as_vertex = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Particle>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![
            0 => Float32x2, // pos
            1 => Float32x2  // vel
        ],
    };
    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("render pipeline"),
        layout: Some(&render_pl_layout),
        vertex: wgpu::VertexState {
            module: &vs,
            entry_point: "vs_main",
            buffers: &[particle_as_vertex],
        },
        fragment: Some(wgpu::FragmentState {
            module: &fs,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: Frame::TEXTURE_FORMAT,
                // blend: Some(wgpu::BlendState::REPLACE),
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::PointList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    // compute layouts and pipelines
    // clear
    let clear_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("clear bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let clear_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("clear layout"),
        bind_group_layouts: &[&clear_bgl],
        push_constant_ranges: &[],
    });
    let clear_counts_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("clear pipeline"),
        layout: Some(&clear_pl_layout),
        module: &cs_clear,
        entry_point: "clear_main",
    });
    let clear_counts_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("clear bg"),
        layout: &clear_bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: grid_counts.as_entire_binding(),
        }],
    });

    // build
    let build_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("build bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let build_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("build layout"),
        bind_group_layouts: &[&build_bgl],
        push_constant_ranges: &[],
    });
    let build_grid_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("build pipeline"),
        layout: Some(&build_pl_layout),
        module: &cs_build,
        entry_point: "build_main",
    });
    let build_bg_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("build A"),
        layout: &build_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buf_a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: grid_counts.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: grid_indices.as_entire_binding(),
            },
        ],
    });
    let build_bg_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("build B"),
        layout: &build_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buf_b.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: grid_counts.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: grid_indices.as_entire_binding(),
            },
        ],
    });

    // params (group 1)
    let params_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("params bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let params_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("params bg"),
        layout: &params_bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: params_buf.as_entire_binding(),
        }],
    });

    // interact
    let interact_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("interact bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let interact_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("interact layout"),
        bind_group_layouts: &[&interact_bgl, &params_bgl], // group(0)=data, group(1)=params
        push_constant_ranges: &[],
    });
    let interact_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("interact pipeline"),
        layout: Some(&interact_pl_layout),
        module: &cs_interact,
        entry_point: "interact_main",
    });
    let interact_bg_a2b = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("interact A->B"),
        layout: &interact_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buf_a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buf_b.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: grid_counts.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: grid_indices.as_entire_binding(),
            },
        ],
    });
    let interact_bg_b2a = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("interact B->A"),
        layout: &interact_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buf_b.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buf_a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: grid_counts.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: grid_indices.as_entire_binding(),
            },
        ],
    });

    Model {
        render_pipeline,
        clear_counts_pipeline,
        build_grid_pipeline,
        interact_pipeline,
        buf_a,
        buf_b,
        source_is_a: true,
        grid_counts,
        grid_indices,
        clear_counts_bg,
        build_bg_a,
        build_bg_b,
        interact_bg_a2b,
        interact_bg_b2a,
        params_buf,
        params_bg,
    }
}

fn update(app: &App, m: &mut Model, _u: Update) {
    m.source_is_a = !m.source_is_a;

    // mouse → NDC
    let win = app.main_window().rect();
    let mp = app.mouse.position();
    let ndc_x = (mp.x / (win.w() * 0.5)) as f32;
    let ndc_y = (mp.y / (win.h() * 0.5)) as f32;

    let params = Params {
        mouse: [ndc_x.clamp(-1.0, 1.0), ndc_y.clamp(-1.0, 1.0)],
        radius2: 0.05,    // tweak at runtime if you like
        strength: 0.001, // tweak
    };
    let window = app.main_window();
    let queue = window.queue();
    queue.write_buffer(&m.params_buf, 0, bytemuck::bytes_of(&params));
}

fn view(app: &App, m: &Model, frame: Frame) {
    // CPU fade layer
    let draw = app.draw();
    draw.rect()
        .wh(app.window_rect().wh())
        .rgba(0.0, 0.0, 0.0, 0.03);
    draw.to_frame(app, &frame).unwrap();

    let mut enc = frame.command_encoder();

    // choose groups
    let (build_bg, interact_bg, render_buf) = if m.source_is_a {
        (&m.build_bg_a, &m.interact_bg_a2b, &m.buf_b)
    } else {
        (&m.build_bg_b, &m.interact_bg_b2a, &m.buf_a)
    };

    // clear
    {
        let mut c = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("clear_counts"),
        });
        c.set_pipeline(&m.clear_counts_pipeline);
        c.set_bind_group(0, &m.clear_counts_bg, &[]);
        let groups = (NUM_CELLS + 255) / 256;
        c.dispatch_workgroups(groups, 1, 1);
    }

    // build grid
    {
        let mut c = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("build_grid"),
        });
        c.set_pipeline(&m.build_grid_pipeline);
        c.set_bind_group(0, build_bg, &[]);
        let groups = ((N_PARTICLES as u32) + 255) / 256;
        c.dispatch_workgroups(groups, 1, 1);
    }

    // interact + mouse repel
    {
        let mut c = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("interact"),
        });
        c.set_pipeline(&m.interact_pipeline);
        c.set_bind_group(0, interact_bg, &[]);
        c.set_bind_group(1, &m.params_bg, &[]); // mouse params
        let groups = ((N_PARTICLES as u32) + 255) / 256;
        c.dispatch_workgroups(groups, 1, 1);
    }

    // render
    {
        let mut r = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("render"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: frame.texture_view(),
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: true,
                },
            })],
            depth_stencil_attachment: None,
        });
        r.set_pipeline(&m.render_pipeline);
        r.set_vertex_buffer(0, render_buf.slice(..));
        r.draw(0..N_PARTICLES as u32, 0..1);
    }
}
