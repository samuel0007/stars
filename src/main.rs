use bytemuck::{Pod, Zeroable};
use nannou::Frame;
use nannou::prelude::*;
use nannou::wgpu::{self, util::DeviceExt};

// ---------- constants ----------
const N_PARTICLES: usize = 10_000;
const GRID_SIZE: u32 = 64;
const MAX_PER_CELL: u32 = 64;
const NUM_CELLS: u32 = GRID_SIZE * GRID_SIZE;

const MOUSE_RADIUS2: f32 = 0.05;
const MOUSE_STRENGTH: f32 = 0.0005; // 0.0015
const FORCE_RADIUS2: f32 = 0.002;
const FORCE_STRENGTH: f32 = 0.000000002;
const PARTICLE_DAMPING: f32 = 0.99;
const DT: f32 = 1.;

const V0: f32 = 0.0000001; // init particle vel

const WGS: u32 = 256; // work group size for compute shaders

// ---------- data ----------
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Particle {
    pos: [f32; 2],
    vel: [f32; 2],
    mass: f32,
    prefix: u32,
    pad: [u32; 2], // padding for 16B alignment 32 + 16 = 48
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct MortonPair {
    code: u32,
    idx: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params {
    mouse: [f32; 2],
    mouse_radius2: f32,
    mouse_strength: f32,
    force_radius2: f32,
    force_strength: f32,
    particle_damping: f32,
    dt: f32,
    level: u32,
    _pad: u32,
}

// ---------- model ----------
struct Pipelines {
    render: wgpu::RenderPipeline,
    clear_counts: wgpu::ComputePipeline,
    build_grid: wgpu::ComputePipeline,
    interact: wgpu::ComputePipeline,
    morton: wgpu::ComputePipeline,
}

struct Buffers {
    a: wgpu::Buffer,
    b: wgpu::Buffer,
    grid_counts: wgpu::Buffer,
    grid_indices: wgpu::Buffer,
    params: wgpu::Buffer,
    morton_buffer: wgpu::Buffer,
}

struct BindGroups {
    clear_counts_bg: wgpu::BindGroup,
    build_a_bg: wgpu::BindGroup,
    build_b_bg: wgpu::BindGroup,
    interact_a2b_bg: wgpu::BindGroup,
    interact_b2a_bg: wgpu::BindGroup,
    params_bg: wgpu::BindGroup,
    morton_a_bg: wgpu::BindGroup,
    morton_b_bg: wgpu::BindGroup,
}

struct Model {
    pipes: Pipelines,
    bufs: Buffers,
    bgs: BindGroups,
    src_is_a: bool,
}

// ---------- app ----------
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

    // --- particles
    let mut init = Vec::with_capacity(N_PARTICLES);
    for _ in 0..N_PARTICLES {
        init.push(Particle {
            pos: [random_range(-1.0, 1.0), random_range(-1.0, 1.0)],
            vel: [random_range(-V0, V0), random_range(-V0, V0)],
            mass: random_range(0.5, 5.),
            prefix: 0,
            pad: [0, 0],
        });
    }

    let morton_init = vec![MortonPair { code: 0, idx: 0 }; N_PARTICLES];

    let a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("particles A"),
        contents: bytemuck::cast_slice(&init),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE,
    });
    let b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("particles B"),
        contents: bytemuck::cast_slice(&init),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE,
    });

    // --- grid
    let grid_counts = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("grid_counts"),
        contents: bytemuck::cast_slice(&vec![0u32; NUM_CELLS as usize]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let grid_indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("grid_indices"),
        contents: bytemuck::cast_slice(&vec![0u32; (NUM_CELLS * MAX_PER_CELL) as usize]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    // --- morton code
    let morton_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("morton buffer"),
        contents: bytemuck::cast_slice(&morton_init),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    // --- params
    let params_init = Params {
        mouse: [0.0, 0.0],
        mouse_radius2: MOUSE_RADIUS2,
        mouse_strength: MOUSE_STRENGTH,
        force_radius2: FORCE_RADIUS2,
        force_strength: FORCE_STRENGTH,
        dt: DT,
        particle_damping: PARTICLE_DAMPING,
        level: 0,
        _pad: 0,
    };
    let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::bytes_of(&params_init),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // --- shaders
    let vs = shader(&device, "vs", shaders::vs());
    let fs = shader(&device, "fs", shaders::fs(GRID_SIZE));
    let cs_clear: wgpu::ShaderModule = shader(&device, "cs_clear", shaders::cs_clear(GRID_SIZE));
    let cs_build = shader(
        &device,
        "cs_build",
        shaders::cs_build(GRID_SIZE, MAX_PER_CELL),
    );
    let cs_interact = shader(
        &device,
        "cs_interact",
        shaders::cs_interact(GRID_SIZE, MAX_PER_CELL),
    );
    let cs_morton = shader(&device, "cs_morton", shaders::cs_morton_code());

    // --- layouts
    let clear_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("clear bgl"),
        entries: &[storage_rw(0)],
    });
    let build_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("build bgl"),
        entries: &[storage_ro(0), storage_rw(1), storage_rw(2)],
    });
    let params_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("params bgl"),
        entries: &[uniform(0)],
    });
    let interact_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("interact bgl"),
        entries: &[storage_ro(0), storage_rw(1), storage_ro(2), storage_ro(3)],
    });
    let morton_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("morton bgl"),
        entries: &[storage_ro(0), storage_rw(1)],
    });

    // --- pipelines
    let render = {
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("render layout"),
            bind_group_layouts: &[&params_bgl],
            push_constant_ranges: &[],
        });
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("render pipeline"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &vs,
                entry_point: "vs_main",
                buffers: &[particle_vbuf_layout()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &fs,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: Frame::TEXTURE_FORMAT,
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
        })
    };
    let clear_counts = {
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("clear layout"),
            bind_group_layouts: &[&clear_bgl],
            push_constant_ranges: &[],
        });
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("clear pipeline"),
            layout: Some(&pl),
            module: &cs_clear,
            entry_point: "clear_main",
        })
    };
    let build_grid = {
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("build layout"),
            bind_group_layouts: &[&build_bgl],
            push_constant_ranges: &[],
        });
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("build pipeline"),
            layout: Some(&pl),
            module: &cs_build,
            entry_point: "build_main",
        })
    };
    let interact = {
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("interact layout"),
            bind_group_layouts: &[&interact_bgl, &params_bgl],
            push_constant_ranges: &[],
        });
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("interact pipeline"),
            layout: Some(&pl),
            module: &cs_interact,
            entry_point: "interact_main",
        })
    };

    let morton = {
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("morton layout"),
            bind_group_layouts: &[&morton_bgl],
            push_constant_ranges: &[],
        });
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("morton pipeline"),
            layout: Some(&pl),
            module: &cs_morton,
            entry_point: "morton_main",
        })
    };

    // --- bind groups
    let clear_counts_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("clear bg"),
        layout: &clear_bgl,
        entries: &[bind_ent(0, &grid_counts)],
    });
    let build_a_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("build A"),
        layout: &build_bgl,
        entries: &[
            bind_ent(0, &a),
            bind_ent(1, &grid_counts),
            bind_ent(2, &grid_indices),
        ],
    });
    let build_b_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("build B"),
        layout: &build_bgl,
        entries: &[
            bind_ent(0, &b),
            bind_ent(1, &grid_counts),
            bind_ent(2, &grid_indices),
        ],
    });
    let interact_a2b_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("interact A->B"),
        layout: &interact_bgl,
        entries: &[
            bind_ent(0, &a),
            bind_ent(1, &b),
            bind_ent(2, &grid_counts),
            bind_ent(3, &grid_indices),
        ],
    });
    let interact_b2a_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("interact B->A"),
        layout: &interact_bgl,
        entries: &[
            bind_ent(0, &b),
            bind_ent(1, &a),
            bind_ent(2, &grid_counts),
            bind_ent(3, &grid_indices),
        ],
    });
    let params_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("params bg"),
        layout: &params_bgl,
        entries: &[bind_ent(0, &params)],
    });
    let morton_a_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("morton a bg"),
        layout: &morton_bgl,
        entries: &[bind_ent(0, &a), bind_ent(1, &morton_buffer)],
    });
    let morton_b_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("morton b bg"),
        layout: &morton_bgl,
        entries: &[bind_ent(0, &b), bind_ent(1, &morton_buffer)],
    });

    Model {
        pipes: Pipelines {
            render,
            clear_counts,
            build_grid,
            interact,
            morton,
        },
        bufs: Buffers {
            a,
            b,
            grid_counts,
            grid_indices,
            params,
            morton_buffer,
        },
        bgs: BindGroups {
            clear_counts_bg,
            build_a_bg,
            build_b_bg,
            interact_a2b_bg,
            interact_b2a_bg,
            params_bg,
            morton_a_bg,
            morton_b_bg,
        },
        src_is_a: true,
    }
}

fn update(app: &App, m: &mut Model, _u: Update) {
    m.src_is_a = !m.src_is_a;

    // mouse → NDC
    let win = app.main_window().rect();
    let mp = app.mouse.position();
    let ndc = [
        (mp.x / (win.w() * 0.5)).clamp(-1.0, 1.0) as f32,
        (mp.y / (win.h() * 0.5)).clamp(-1.0, 1.0) as f32,
    ];

    // TODO make these interactive params
    let params = Params {
        mouse: ndc,
        mouse_radius2: MOUSE_RADIUS2,
        mouse_strength: MOUSE_STRENGTH,
        force_radius2: FORCE_RADIUS2,
        force_strength: FORCE_STRENGTH,
        dt: DT,
        particle_damping: PARTICLE_DAMPING,
        level: 0,
        _pad: 0,
    };
    let window = app.main_window();
    let q = window.queue();
    q.write_buffer(&m.bufs.params, 0, bytemuck::bytes_of(&params));
}

fn view(app: &App, m: &Model, frame: Frame) {
    // fade layer
    let draw = app.draw();
    draw.rect()
        .wh(app.window_rect().wh())
        .rgba(0.0, 0.0, 0.0, 0.03);
    draw.to_frame(app, &frame).unwrap();

    let mut enc = frame.command_encoder();

    // select ping-pong route once
    let (build_bg, interact_bg, morton_bg, render_buf) = if m.src_is_a {
        (
            &m.bgs.build_a_bg,
            &m.bgs.interact_a2b_bg,
            &m.bgs.morton_a_bg,
            &m.bufs.b,
        )
    } else {
        (
            &m.bgs.build_b_bg,
            &m.bgs.interact_b2a_bg,
            &m.bgs.morton_b_bg,
            &m.bufs.a,
        )
    };

    // clear counts
    {
        let mut c = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("clear_counts"),
        });
        c.set_pipeline(&m.pipes.clear_counts);
        c.set_bind_group(0, &m.bgs.clear_counts_bg, &[]);
        c.dispatch_workgroups(div_ceil(NUM_CELLS, WGS), 1, 1);
    }

    // build grid
    {
        let mut c = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("build_grid"),
        });
        c.set_pipeline(&m.pipes.build_grid);
        c.set_bind_group(0, build_bg, &[]);
        c.dispatch_workgroups(div_ceil(N_PARTICLES as u32, WGS), 1, 1);
    }

    // interact
    {
        let mut c = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("interact"),
        });
        c.set_pipeline(&m.pipes.interact);
        c.set_bind_group(0, interact_bg, &[]);
        c.set_bind_group(1, &m.bgs.params_bg, &[]);
        c.dispatch_workgroups(div_ceil(N_PARTICLES as u32, WGS), 1, 1);
    }

    // Compute morton code
    {
        let mut c = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("morton"),
        });
        c.set_pipeline(&m.pipes.morton);
        c.set_bind_group(0, morton_bg, &[]);
        c.dispatch_workgroups(div_ceil(N_PARTICLES as u32, WGS), 1, 1);
    }

    // render
    {
        let mut r: wgpu::RenderPass<'_> = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
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
        r.set_pipeline(&m.pipes.render);
        r.set_bind_group(0, &m.bgs.params_bg, &[]);
        r.set_vertex_buffer(0, render_buf.slice(..));
        r.draw(0..N_PARTICLES as u32, 0..1);
    }
}

// ---------- helpers ----------
fn shader(device: &wgpu::Device, label: &str, src: String) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    })
}

const PARTICLE_ATTRS: [wgpu::VertexAttribute; 4] =
    wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Float32, 3 => Uint32];
fn particle_vbuf_layout() -> wgpu::VertexBufferLayout<'static> {
    return wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Particle>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &PARTICLE_ATTRS,
    };
}
fn div_ceil(n: u32, d: u32) -> u32 {
    (n + d - 1) / d
}
fn storage_ro(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
fn storage_rw(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
fn uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
fn bind_ent(binding: u32, buf: &wgpu::Buffer) -> wgpu::BindGroupEntry {
    wgpu::BindGroupEntry {
        binding,
        resource: buf.as_entire_binding(),
    }
}

// ---------- shaders ----------
mod shaders {
    pub fn vs() -> String {
        r#"
struct VsOut {
  @builtin(position) pos : vec4<f32>,
  @location(0) vel : vec2<f32>,
  @location(1) prefix: u32,
};
@vertex
fn vs_main(@location(0) pos: vec2<f32>, @location(1) vel: vec2<f32>, @location(2) _mass: f32, @location(3) prefix: u32) -> VsOut {
  var out: VsOut;
  out.pos = vec4<f32>(pos, 0.0, 1.0);
  out.vel = vel;
  out.prefix = prefix;
  return out;
}
"#
        .into()
    }

    pub fn fs(grid: u32) -> String {
        format!(
            r#"
const GRID_SIZE : u32 = {grid}u;

struct Params {{
  mouse: vec2<f32>,
  mouse_radius2: f32,
  mouse_strength: f32,
  force_radius2: f32,
  force_strength: f32,
  particle_damping: f32,
  dt: f32,
  level: u32,
  _pad: u32,
}};

@group(0) @binding(0) var<uniform> params: Params;

@fragment
fn fs_main(@location(0) vel: vec2<f32>, @location(1) prefix: u32) -> @location(0) vec4<f32> {{
    // quadtree depth from grid size (e.g. 64 -> 6)
    let depth = u32(log2(f32(GRID_SIZE)));
    let lvl = min(params.level, depth);

    // top-level quadrant = the highest morton pair (yMSB,xMSB)
    let top_pair = (prefix >> (2u * depth - 2u)) & 3u;

    if (lvl == 1u) {{
        // Fixed palette: 4 distinct colors for the 4 top-level quadrants
        // 0: NE (depending on your morton ordering), 1,2,3 likewise
        var c: vec3<f32>;
        switch (top_pair) {{
            case 0u: {{ c = vec3<f32>(1.0, 0.25, 0.25); }} // red-ish
            case 1u: {{ c = vec3<f32>(0.25, 1.0, 0.35); }} // green-ish
            case 2u: {{ c = vec3<f32>(0.30, 0.55, 1.0); }} // blue-ish
            default: {{ c = vec3<f32>(1.0, 0.95, 0.30); }} // yellow-ish
        }}
        return vec4<f32>(c, 1.0);
    }} else {{
        // General case: color by cell at requested level
        let cell_id = prefix >> (2u * (depth - lvl));

        // Simple 32-bit LCG hash to RGB
        let h = cell_id * 1664525u + 1013904223u;
        let r = f32((h        ) & 255u) / 255.0;
        let g = f32((h >>  8u ) & 255u) / 255.0;
        let b = f32((h >> 16u ) & 255u) / 255.0;
        return vec4<f32>(r, g, b, 1.0);
    }}
}}
"#,
            grid = grid
        )
    }

    pub fn cs_clear(grid: u32) -> String {
        format!(
            r#"
const GRID_SIZE : u32 = {grid}u;
struct Counts {{ data: array<atomic<u32>> }};
@group(0) @binding(0) var<storage, read_write> grid_counts : Counts;

@compute @workgroup_size(256)
fn clear_main(@builtin(global_invocation_id) id : vec3<u32>) {{
  let num_cells = GRID_SIZE * GRID_SIZE;
  let i = id.x;
  if (i < num_cells) {{
    atomicStore(&grid_counts.data[i], 0u);
  }}
}}
"#
        )
    }

    pub fn cs_build(grid: u32, max_per_cell: u32) -> String {
        format!(
            r#"
const GRID_SIZE    : u32 = {grid}u;
const MAX_PER_CELL : u32 = {mpc}u;

struct Particle {{
  pos    : vec2<f32>,
  vel    : vec2<f32>,
  mass   : f32,
  prefix : u32,
  pad    : vec2<u32>,
}};
struct Particles {{ data: array<Particle> }};
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
    let pos = particles.data[i].pos;
    let ci = cell_index(pos);
    let offset = atomicAdd(&grid_counts.data[ci], 1u);
    if (offset < MAX_PER_CELL) {{
        let idx = ci * MAX_PER_CELL + offset;
        grid_indices.data[idx] = i;
    }}
}}
"#,
            mpc = max_per_cell
        )
    }

    pub fn cs_interact(grid: u32, max_per_cell: u32) -> String {
        format!(
            r#"
const GRID_SIZE    : u32 = {grid}u;
const MAX_PER_CELL : u32 = {mpc}u;

const EPS     : f32 = 1e-6;

struct Particle {{
  pos    : vec2<f32>,
  vel    : vec2<f32>,
  mass   : f32,
  prefix : u32,
  pad    : vec2<u32>,
}};
struct Particles {{ data: array<Particle> }};
struct Counts    {{ data: array<atomic<u32>> }};
struct Grid      {{ data: array<u32> }};

struct Params {{
  mouse    : vec2<f32>,
  mouse_radius2  : f32,
  mouse_strength : f32,
  force_radius2  : f32,
  force_strength : f32,
  particle_damping : f32,
  dt: f32,
  level: u32,
  _pad: u32,
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

// ---- Morton helpers ----
fn part1by1(n: u32) -> u32 {{
    var x = n & 0x0000ffffu;
    x = (x | (x << 8u)) & 0x00FF00FFu;
    x = (x | (x << 4u)) & 0x0F0F0F0Fu;
    x = (x | (x << 2u)) & 0x33333333u;
    x = (x | (x << 1u)) & 0x55555555u;
    return x;
}}

fn morton2D(x: u32, y: u32) -> u32 {{
    return (part1by1(y) << 1u) | part1by1(x);
}}

@compute @workgroup_size(256)
fn interact_main(@builtin(global_invocation_id) id : vec3<u32>) {{
    let i = id.x;
    if (i >= arrayLength(&in_particles.data)) {{ return; }}

    var pos = in_particles.data[i].pos;
    var vel = in_particles.data[i].vel;
    let mass = in_particles.data[i].mass;

    let dm  = pos - params.mouse;       // repulsion
    let d2m = dot(dm, dm);
    if (d2m < params.mouse_radius2) {{
        let inv = inverseSqrt(d2m + EPS);
        let dir = dm * inv;
        vel += params.mouse_strength * dir;
    }}

    let gc = grid_coord(pos);
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
                let q_pos = in_particles.data[pid].pos;
                let q_mass = in_particles.data[pid].mass;
                let d = q_pos - pos;
                let r2 = dot(d, d);
                if (r2 > EPS && r2 < params.force_radius2) {{
                    let inv_r = inverseSqrt(r2);
                    vel -= params.force_strength * d * inv_r * inv_r * inv_r * params.dt * q_mass;
                }}
            }}
        }}
    }}

    vel *= params.particle_damping;
    pos += vel * params.dt;

    if (pos.x > 1.0 || pos.x < -1.0) {{ vel.x = -vel.x; }}
    if (pos.y > 1.0 || pos.y < -1.0) {{ vel.y = -vel.y; }}

    out_particles.data[i].pos = pos;
    out_particles.data[i].vel = vel;

    // Morton Code Computation for next iteration
    let gx_max = GRID_SIZE - 1u;
    let gy_max = GRID_SIZE - 1u;
    let gx = clamp(u32(((pos.x + 1.0) * 0.5) * f32(GRID_SIZE)), 0u, gx_max);
    let gy = clamp(u32(((pos.y + 1.0) * 0.5) * f32(GRID_SIZE)), 0u, gy_max);
    out_particles.data[i].prefix = morton2D(gx, gy);
}}
"#,
            mpc = max_per_cell
        )
    }

    pub fn cs_morton_code() -> String {
        format!(
            r#"
        const GRID_SIZE : u32 = 1024u;   // adjust to your grid resolution

        struct Particle {{
        pos    : vec2<f32>,
        vel    : vec2<f32>,
        mass   : f32,
        prefix : u32,
        pad    : vec2<u32>,
        }};

        struct Particles {{ data: array<Particle> }};

        struct Morton {{
        code : u32,
        idx  : u32,
        }};

        struct MortonBuf {{ data: array<Morton> }};

        @group(0) @binding(0) var<storage, read>  particles : Particles;
        @group(0) @binding(1) var<storage, write> morton_out : MortonBuf;

        // ---- Morton helpers ----
        fn part1by1(n: u32) -> u32 {{
            var x = n & 0x0000ffffu;
            x = (x | (x << 8u)) & 0x00FF00FFu;
            x = (x | (x << 4u)) & 0x0F0F0F0Fu;
            x = (x | (x << 2u)) & 0x33333333u;
            x = (x | (x << 1u)) & 0x55555555u;
            return x;
        }}

        fn morton2D(x: u32, y: u32) -> u32 {{
            return (part1by1(y) << 1u) | part1by1(x);
        }}

        // ---- Main ----
        @compute @workgroup_size(256)
        fn morton_main(@builtin(global_invocation_id) gid : vec3<u32>) {{
            let i = gid.x;
            if (i >= arrayLength(&particles.data)) {{
                return;
            }}

            // map from [-1,1] NDC → [0, GRID_SIZE-1] integer coords
            let pos = particles.data[i].pos;
            let gx_max = GRID_SIZE - 1u;
            let gy_max = GRID_SIZE - 1u;
            let gx = clamp(u32(((pos.x + 1.0) * 0.5) * f32(GRID_SIZE)), 0u, gx_max);
            let gy = clamp(u32(((pos.y + 1.0) * 0.5) * f32(GRID_SIZE)), 0u, gy_max);
            let code = morton2D(gx, gy);

            morton_out.data[i].code = code;
            morton_out.data[i].idx  = i;
        }}"#,
        )
    }
}
