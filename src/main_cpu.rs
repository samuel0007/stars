use nannou::prelude::*;

const N_PARTICLES: usize = 1_000;

struct Model {
    positions: Vec<Point2>,
    velocities: Vec<Vec2>,
}

fn main() {
    nannou::app(model).update(update).view(view).run();
}

fn model(app: &App) -> Model {
    app.new_window()
        .size(800, 800)
        .title("100k CPU-side rect draw calls")
        .build()
        .unwrap();

    let mut positions = Vec::with_capacity(N_PARTICLES);
    let mut velocities = Vec::with_capacity(N_PARTICLES);

    for _ in 0..N_PARTICLES {
        positions.push(pt2(
            random_range(-400.0, 400.0),
            random_range(-400.0, 400.0),
        ));
        velocities.push(vec2(
            1.,
            1.
            // random_range(-1.0, 1.0) * 0.5,
            // random_range(-1.0, 1.0) * 0.5,
        ));
    }

    Model { positions, velocities }
}

fn update(_app: &App, model: &mut Model, _update: Update) {
    for (pos, vel) in model.positions.iter_mut().zip(model.velocities.iter_mut()) {
        pos.x += vel.x;
        pos.y += vel.y;

        // Bounce off walls
        if pos.x > 400.0 || pos.x < -400.0 {
            vel.x *= -1.0;
        }
        if pos.y > 400.0 || pos.y < -400.0 {
            vel.y *= -1.0;
        }
    }
}


fn view(app: &App, model: &Model, frame: Frame) {
    let draw = app.draw();
    draw.background().color(BLACK);

    // Draw all particles as tiny rects
    for &pos in &model.positions {
        draw.rect()
            .xy(pos)
            .w_h(2.0, 2.0)
            .color(WHITE);
    }

    draw.to_frame(app, &frame).unwrap();
}
