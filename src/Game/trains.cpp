#include "trains.h"
#include <iostream>

Train::Train(Terrain *terrain_grid, int cargo_count, float cell_size, create_mesh_func create_mesh) {
    terrain = terrain_grid;
    cell = cell_size;

    wagons = cargo_count;

    std::vector<VertexFormat> vertices;
    std::vector<uint32_t> indices;

    in_tunnel = false;
    train_moved = false;
    delta_time = 0.0f;

    create_locomotive(vertices, indices);
    train_model.progress = 0.0f;
    train_model.cell_pos = (*terrain_grid->get_stations())[0];
    train_model.orientation = NONE;
    train_model.next_move = NONE;
    distance_to_camera = 20.0f;

    init(train_model.cell_pos, create_mesh);

    create_mesh("train", vertices, indices);
}

void Train::draw(draw_mesh_func draw_mesh) {
    glm::mat4 model = train_model.get_model(terrain, 1.0f);
    draw_mesh("train", model, 1.0f);

    for (int i = 0; i < wagons; i++) {
        glm::mat4 cargo_model = cargos[i].get_model(terrain, 1.0f);
        draw_mesh(("wagon_" + std::to_string(i)).c_str(), cargo_model, 1.0f);
    }
}

void Train::draw_outline(draw_mesh_func draw_mesh,  float scale) {
    glm::mat4 model = train_model.get_model(terrain, scale);
    draw_mesh("train", model, 1.0f);

    for (int i = 0; i < wagons; i++) {
        glm::mat4 cargo_model = cargos[i].get_model(terrain, scale);
        draw_mesh(("wagon_" + std::to_string(i)).c_str(), cargo_model, 1.0f);
    }
}

void Train::create_locomotive(std::vector<VertexFormat> &vertices, std::vector<uint32_t> &indices)
{
    glm::vec3 locomotiveColor = glm::vec3(0.2f, 0.2f, 0.8f);  // Blue
    glm::vec3 wheelColor = glm::vec3(0.8f, 0.1f, 0.1f);       // Red
    glm::vec3 cabinColor = glm::vec3(0.1f, 0.7f, 0.3f);
    glm::vec3 platformColor = glm::vec3(1.0f, 0.984f, 0.05f);
    
    float wheelRadius = 0.1f * cell;
    float wheelWidth = 0.05f * cell;

    uint32_t currentIndex = 0;
    float len = 0.9f;
    float t = 0.40f;
    float margin = 0.02f * cell;

    float cabinWidth = t * cell * len;
    float cabinHeight = 1.8f;
    float cabinDepth = 0.4f * cell;
    float cabinX = 0.0f;
    float cabinY = wheelRadius * 2.0f - (0.001f * cell);
    float cabinZ = -cabinDepth / 2.0f;
    
    create_box(vertices, indices, currentIndex,
               cabinX, cabinY + 0.1f, cabinZ,
               cabinWidth, cabinHeight, cabinDepth,
               cabinColor);
    
    create_box(vertices, indices, currentIndex,
               cabinX - margin, cabinY, cabinZ - margin,
               len * cell + 2.0f * margin, 0.05f * cell, cabinDepth + 2.0f * margin,
               platformColor);
    
    float engineRadius = 0.18f * cell;
    float engineLength = (1.0f - t) * cell * len;
    float engineX = cabinX + cabinWidth;
    float engineY = cabinY + engineRadius;
    float engineZ = 0.0f;
    
    create_cylinder(vertices, indices, currentIndex,
                   engineX, engineY, engineZ,
                   engineRadius, engineLength,
                   locomotiveColor, 20, 'X');
    
    int numLocomotiveWheels = std::floor((len * cell) / (2.0f * wheelRadius));
    float wheelSpacing = (len * cell - numLocomotiveWheels * (2.0f * wheelRadius)) / (numLocomotiveWheels - 1.0f);
    
    for (int i = 0; i < numLocomotiveWheels; i++) {
        float wheelX = cabinX + wheelRadius + i * (2.0f * wheelRadius) + (i > 0 ? (i - 1) * wheelSpacing : 0.0f);
        float wheelY = wheelRadius;
        float wheelZ = cabinDepth / 2.0f - 0.05f * cell;
        
        create_cylinder(vertices, indices, currentIndex,
                       wheelX, wheelY, +wheelZ,
                       wheelRadius, wheelWidth,
                       wheelColor, 16, 'Z');

        create_cylinder(vertices, indices, currentIndex,
                       wheelX, wheelY, -wheelZ,
                       wheelRadius, wheelWidth,
                       wheelColor, 16, 'Z');
    }
}

void Train::create_wagon(std::vector<VertexFormat> &vertices, std::vector<uint32_t> &indices) {
    glm::vec3 locomotiveColor = glm::vec3(0.2f, 0.2f, 0.8f);  // Blue
    glm::vec3 wheelColor = glm::vec3(0.8f, 0.1f, 0.1f);       // Red
    glm::vec3 cabinColor = glm::vec3(0.1f, 0.7f, 0.3f);
    glm::vec3 platformColor = glm::vec3(1.0f, 0.984f, 0.05f);
    
    float wheelRadius = 0.1f * cell;
    float wheelWidth = 0.05f * cell;

    uint32_t currentIndex = 0;
    float len = 0.9f;
    float t = 1.0f;
    float margin = 0.02f * cell;

    float cabinWidth = t * cell * len;
    float cabinHeight = 1.8f;
    float cabinDepth = 0.4f * cell;
    float cabinX = 0.0f;
    float cabinY = wheelRadius * 2.0f - (0.001f * cell);
    float cabinZ = -cabinDepth / 2.0f;
    
    create_box(vertices, indices, currentIndex,
               cabinX, cabinY + 0.1f, cabinZ,
               cabinWidth, cabinHeight, cabinDepth,
               cabinColor);
    
    create_box(vertices, indices, currentIndex,
               cabinX - margin, cabinY, cabinZ - margin,
               len * cell + 2.0f * margin, 0.05f * cell, cabinDepth + 2.0f * margin,
               platformColor);
    
    int numLocomotiveWheels = 2;
    float wheel_pos[] = {cabinX + wheelRadius + 0.1f * cell, cabinX + len * cell - wheelRadius - 0.1f * cell};
    
    for (int i = 0; i < numLocomotiveWheels; i++) {
        float wheelX = wheel_pos[i];
        float wheelY = wheelRadius;
        float wheelZ = cabinDepth / 2.0f - 0.05f * cell;
        
        create_cylinder(vertices, indices, currentIndex,
                       wheelX, wheelY, +wheelZ,
                       wheelRadius, wheelWidth,
                       wheelColor, 16, 'Z');

        create_cylinder(vertices, indices, currentIndex,
                       wheelX, wheelY, -wheelZ,
                       wheelRadius, wheelWidth,
                       wheelColor, 16, 'Z');
    }
}

void Train::create_box(std::vector<VertexFormat> &vertices, std::vector<uint32_t> &indices,
                       uint32_t &currentIndex, float x, float y, float z,
                       float width, float height, float depth, glm::vec3 color) {
    
    uint32_t base = currentIndex;
    
    glm::vec3 normalFront = glm::vec3(0, 0, -1);
    glm::vec3 normalBack = glm::vec3(0, 0, 1);
    glm::vec3 normalLeft = glm::vec3(-1, 0, 0);
    glm::vec3 normalRight = glm::vec3(1, 0, 0);
    glm::vec3 normalTop = glm::vec3(0, 1, 0);
    glm::vec3 normalBottom = glm::vec3(0, -1, 0);
    
    vertices.push_back(VertexFormat(glm::vec3(x, y, z), color, normalFront));                    // 0
    vertices.push_back(VertexFormat(glm::vec3(x + width, y, z), color, normalFront));            // 1
    vertices.push_back(VertexFormat(glm::vec3(x + width, y + height, z), color, normalFront));   // 2
    vertices.push_back(VertexFormat(glm::vec3(x, y + height, z), color, normalFront));           // 3
    
    vertices.push_back(VertexFormat(glm::vec3(x, y, z + depth), color, normalBack));             // 4
    vertices.push_back(VertexFormat(glm::vec3(x + width, y, z + depth), color, normalBack));     // 5
    vertices.push_back(VertexFormat(glm::vec3(x + width, y + height, z + depth), color, normalBack)); // 6
    vertices.push_back(VertexFormat(glm::vec3(x, y + height, z + depth), color, normalBack));    // 7
    
    vertices.push_back(VertexFormat(glm::vec3(x, y, z), color, normalLeft));                     // 8
    vertices.push_back(VertexFormat(glm::vec3(x, y + height, z), color, normalLeft));            // 9
    vertices.push_back(VertexFormat(glm::vec3(x, y + height, z + depth), color, normalLeft));    // 10
    vertices.push_back(VertexFormat(glm::vec3(x, y, z + depth), color, normalLeft));             // 11
    
    vertices.push_back(VertexFormat(glm::vec3(x + width, y, z), color, normalRight));            // 12
    vertices.push_back(VertexFormat(glm::vec3(x + width, y + height, z), color, normalRight));   // 13
    vertices.push_back(VertexFormat(glm::vec3(x + width, y + height, z + depth), color, normalRight)); // 14
    vertices.push_back(VertexFormat(glm::vec3(x + width, y, z + depth), color, normalRight));    // 15
    
    vertices.push_back(VertexFormat(glm::vec3(x, y + height, z), color, normalTop));             // 16
    vertices.push_back(VertexFormat(glm::vec3(x + width, y + height, z), color, normalTop));     // 17
    vertices.push_back(VertexFormat(glm::vec3(x + width, y + height, z + depth), color, normalTop)); // 18
    vertices.push_back(VertexFormat(glm::vec3(x, y + height, z + depth), color, normalTop));     // 19
    
    vertices.push_back(VertexFormat(glm::vec3(x, y, z), color, normalBottom));                   // 20
    vertices.push_back(VertexFormat(glm::vec3(x + width, y, z), color, normalBottom));           // 21
    vertices.push_back(VertexFormat(glm::vec3(x + width, y, z + depth), color, normalBottom));   // 22
    vertices.push_back(VertexFormat(glm::vec3(x, y, z + depth), color, normalBottom));           // 23
    
    indices.insert(indices.end(), {base+0, base+3, base+2, base+0, base+2, base+1});
    indices.insert(indices.end(), {base+4, base+5, base+6, base+4, base+6, base+7});
    indices.insert(indices.end(), {base+8, base+11, base+10, base+8, base+10, base+9});
    indices.insert(indices.end(), {base+12, base+13, base+14, base+12, base+14, base+15});
    indices.insert(indices.end(), {base+16, base+19, base+18, base+16, base+18, base+17});
    indices.insert(indices.end(), {base+20, base+21, base+22, base+20, base+22, base+23});
    
    currentIndex += 24;
}

void Train::create_cylinder(std::vector<VertexFormat> &vertices, std::vector<uint32_t> &indices,
                           uint32_t &currentIndex, float x, float y, float z,
                           float radius, float length, glm::vec3 color, int segments, char axis) {
    uint32_t base = currentIndex;
    float start1, start2, end1, end2;
    glm::vec3 normalAxisNeg, normalAxisPos;
    
    if (axis == 'X' || axis == 'x') {
        start1 = x;
        end1 = x + length;
        start2 = end2 = 0;
        normalAxisNeg = glm::vec3(-1, 0, 0);
        normalAxisPos = glm::vec3(1, 0, 0);
    } else {
        start1 = z - length / 2.0f;
        end1 = z + length / 2.0f;
        start2 = end2 = 0;
        normalAxisNeg = glm::vec3(0, 0, -1);
        normalAxisPos = glm::vec3(0, 0, 1);
    }
    
    for (int i = 0; i <= segments; i++) {
        float angle = 2.0f * M_PI * i / segments;
        glm::vec3 pos1, pos2, normal;
        
        if (axis == 'X' || axis == 'x') {
            float cy = y + radius * cos(angle);
            float cz = z + radius * sin(angle);
            pos1 = glm::vec3(start1, cy, cz);
            pos2 = glm::vec3(end1, cy, cz);
            normal = glm::normalize(glm::vec3(0, cos(angle), sin(angle)));
        } else {
            float cx = x + radius * cos(angle);
            float cy = y + radius * sin(angle);
            pos1 = glm::vec3(cx, cy, start1);
            pos2 = glm::vec3(cx, cy, end1);
            normal = glm::normalize(glm::vec3(cos(angle), sin(angle), 0));
        }
        
        vertices.push_back(VertexFormat(pos1, color, normal));
        vertices.push_back(VertexFormat(pos2, color, normal));
    }
    
    for (int i = 0; i < segments; i++) {
        uint32_t i0 = base + i * 2;
        uint32_t i1 = base + i * 2 + 1;
        uint32_t i2 = base + (i + 1) * 2;
        uint32_t i3 = base + (i + 1) * 2 + 1;
        
        indices.insert(indices.end(), {i0, i2, i1});
        indices.insert(indices.end(), {i1, i2, i3});
    }
    
    currentIndex += (segments + 1) * 2;

    uint32_t startCapBase = currentIndex;
    glm::vec3 centerStart = (axis == 'X' || axis == 'x') ? 
                            glm::vec3(start1, y, z) : 
                            glm::vec3(x, y, start1);
    vertices.push_back(VertexFormat(centerStart, color, normalAxisNeg));
    
    for (int i = 0; i <= segments; i++) {
        float angle = 2.0f * M_PI * i / segments;
        glm::vec3 pos;
        
        if (axis == 'X' || axis == 'x') {
            float cy = y + radius * cos(angle);
            float cz = z + radius * sin(angle);
            pos = glm::vec3(start1, cy, cz);
        } else {
            float cx = x + radius * cos(angle);
            float cy = y + radius * sin(angle);
            pos = glm::vec3(cx, cy, start1);
        }
        
        vertices.push_back(VertexFormat(pos, color, normalAxisNeg));
    }
    
    uint32_t startCenter = startCapBase;
    for (int i = 0; i < segments; i++) {
        indices.insert(indices.end(), {startCenter, startCapBase + i + 2, startCapBase + i + 1});
    }
    
    currentIndex += segments + 2;

    uint32_t endCapBase = currentIndex;
    glm::vec3 centerEnd = (axis == 'X' || axis == 'x') ? 
                          glm::vec3(end1, y, z) : 
                          glm::vec3(x, y, end1);
    vertices.push_back(VertexFormat(centerEnd, color, normalAxisPos));
    
    for (int i = 0; i <= segments; i++) {
        float angle = 2.0f * M_PI * i / segments;
        glm::vec3 pos;
        
        if (axis == 'X' || axis == 'x') {
            float cy = y + radius * cos(angle);
            float cz = z + radius * sin(angle);
            pos = glm::vec3(end1, cy, cz);
        } else {
            float cx = x + radius * cos(angle);
            float cy = y + radius * sin(angle);
            pos = glm::vec3(cx, cy, end1);
        }
        
        vertices.push_back(VertexFormat(pos, color, normalAxisPos));
    }
    
    uint32_t endCenter = endCapBase;
    for (int i = 0; i < segments; i++) {
        indices.insert(indices.end(), {endCenter, endCapBase + i + 1, endCapBase + i + 2});
    }
    
    currentIndex += segments + 2;
}

glm::mat4 basic_model::get_model(Terrain *terrain, float scale)
{
    float t = glm::clamp(progress, 0.0f, 1.0f);
    glm::ivec2 p = cell_pos;

    grid_cell c0 = terrain->get_on_train_grid(p.x, p.y);
    if (c0.height < 0.0f)
        return glm::mat4(1.0f);

    glm::ivec2 dir(0,0);
    float yaw = 0.0f;
    switch (orientation) {
        case MOVE_W: dir={0,-1}; yaw= M_PI_2;  break;
        case MOVE_S: dir={0, 1}; yaw=-M_PI_2;  break;
        case MOVE_A: dir={-1,0}; yaw= M_PI;    break;
        case MOVE_D: dir={ 1,0}; yaw= 0.0f;    break;
        default: break;
    }

    glm::ivec2 np = p + dir;
    grid_cell c1 = terrain->get_on_train_grid(np.x, np.y);
    if (c1.height < 0.0f) {
        c1 = c0;
        t = 0.0f;
    }

    float h = glm::mix(c0.height, c1.height, t) + 0.5f;
    glm::vec3 n = glm::normalize(glm::mix(c0.norm, c1.norm, t));

    glm::vec2 basePos(
        p.x * GRID_CELL_SIZE + 0.5f * GRID_CELL_SIZE,
        p.y * GRID_CELL_SIZE + 0.5f * GRID_CELL_SIZE
    );

    glm::vec2 moveDelta = glm::vec2(dir) * (t * GRID_CELL_SIZE);
    glm::vec2 worldPos = basePos + moveDelta;

    glm::mat4 T = glm::translate(glm::mat4(1.0f),
                                 glm::vec3(worldPos.x, h, worldPos.y));

    glm::vec3 up(0,1,0);
    float dotVal = glm::clamp(glm::dot(up, n), -1.0f, 1.0f);
    float angle = std::acos(dotVal);
    glm::vec3 axis = glm::cross(up, n);

    glm::mat4 R_tilt = (glm::length(axis) < 1e-6f)
                       ? glm::mat4(1.0f)
                       : glm::rotate(glm::mat4(1.0f), angle, glm::normalize(axis));

    glm::mat4 R_yaw = glm::rotate(glm::mat4(1.0f), yaw, glm::vec3(0,1,0));

    glm::mat4 backShift =
        glm::translate(glm::mat4(1.0f),
                       glm::vec3(-0.5f * GRID_CELL_SIZE, 0.0f, 0.0f));

    glm::mat4 scale_mat = glm::scale(glm::mat4(1.0f), glm::vec3(scale));

    glm::mat4 model = T * R_tilt * R_yaw * backShift * scale_mat;
    return model;
}

const float TRAIN_SPEED = 30.0f;


static std::vector<MOVE> get_valid_neighbors(Terrain* terrain, const glm::ivec2 &pos, MOVE current_orientation) {
    std::vector<MOVE> neighbors;

    if (terrain->get_on_train_grid(pos.x, pos.y - 1).height >= 0 && current_orientation != MOVE_S) neighbors.push_back(MOVE_W);
    if (terrain->get_on_train_grid(pos.x, pos.y + 1).height >= 0 && current_orientation != MOVE_W) neighbors.push_back(MOVE_S);
    if (terrain->get_on_train_grid(pos.x + 1, pos.y).height >= 0 && current_orientation != MOVE_A) neighbors.push_back(MOVE_D);
    if (terrain->get_on_train_grid(pos.x - 1, pos.y).height >= 0 && current_orientation != MOVE_D) neighbors.push_back(MOVE_A);

    return neighbors;
}

MOVE opposite(MOVE m) {
    switch (m) {
        case MOVE_S: return MOVE_W;
        case MOVE_W: return MOVE_S;
        case MOVE_D: return MOVE_A;
        case MOVE_A: return MOVE_D;
        default: return NONE;
    }
}

void Train::init(glm::ivec2 start, create_mesh_func create_mesh) {
    train_model.cell_pos = start;
    train_model.progress = 0.0f;
    train_model.next_move = NONE;

    train_model.orientation = NONE;
    std::vector<MOVE> neighbors = get_valid_neighbors(terrain, start, NONE);

    if (!neighbors.empty()) {
        train_model.orientation = neighbors[0];
    }

    cargos.clear();

    // opposite of locomotive orientation
    MOVE backDir = opposite(train_model.orientation);
    glm::ivec2 dir(0, 0);
    switch (backDir) {
        case MOVE_W: dir = {0, -1}; break;
        case MOVE_S: dir = {0,  1}; break;
        case MOVE_A: dir = {-1, 0}; break;
        case MOVE_D: dir = { 1, 0}; break;
        default: break;
    }

    for (int i = 0; i < wagons; ++i) {
        basic_model wagon;
        wagon.progress = 0.0f;
        wagon.orientation = train_model.orientation;
        wagon.next_move = NONE;

        std::vector<VertexFormat> vertices;
        std::vector<uint32_t> indices;

        create_wagon(vertices, indices);
        create_mesh(("wagon_" + std::to_string(i)).c_str(), vertices, indices);

        wagon.cell_pos = train_model.cell_pos + dir * (i + 1);

        cargos.push_back(wagon);
    }
}

void Train::update(float _delta_time, implemented::Camera *camera)
{   
    glm::vec3 camera_pivot = train_model.get_pos(terrain);
    camera->position = camera_pivot;
    camera->distanceToTarget = distance_to_camera;
    camera->position -= camera->forward * distance_to_camera;

    if (train_model.orientation == NONE || terrain->is_gameover()) return;

    delta_time = _delta_time;

    float cell_distance = cell;

    train_moved = false;
    glm::ivec2 next_pos = train_model.cell_pos;
    switch (train_model.orientation) {
        case MOVE_W: next_pos.y -= 1; break;
        case MOVE_S: next_pos.y += 1; break;
        case MOVE_D: next_pos.x += 1; break;
        case MOVE_A: next_pos.x -= 1; break;
        default: break;
    }

    in_tunnel = terrain->is_tunnel_cell(next_pos.x, next_pos.y);

    if (train_model.progress >= 1.0f) {
        train_moved = false;

        auto neighbors = get_valid_neighbors(terrain, next_pos, train_model.orientation);
        //std::cout << neighbors.size() << std::endl;

        bool is_intersection = neighbors.size() >= 2;

        if (is_intersection) {
            if (train_model.next_move != NONE &&
                std::find(neighbors.begin(), neighbors.end(), train_model.next_move) != neighbors.end()) {
                train_model.orientation = train_model.next_move;

                for (int i = 0; i < wagons; i++)
                    cargos[i].train_movements.push(train_model.next_move);

                train_model.next_move = NONE;
                button_pressed = NONE;
            } else {
                //std::cout << "waiting... " << train_model.progress << std::endl;
                train_model.progress = 1.0f;
                return;
            }
        } else {
            train_model.orientation = neighbors[0];
        }

        train_model.cell_pos = next_pos;
        train_model.progress -= 1.0f;
    } else {
        train_model.progress += (TRAIN_SPEED * delta_time) / cell_distance;
        train_moved = true;
    }

    update_wagons();

    terrain->handle_train(train_model.cell_pos);
}

void Train::update_wagons() {
    if (cargos.empty()) return;

    float cell_distance = cell; // GRID_CELL_SIZE

    for (int i = 0; i < wagons; i++) {
        auto& wagon = cargos[i];

        if (wagon.progress >= 1.0f) {
            glm::ivec2 next_pos = wagon.cell_pos;
            switch (wagon.orientation) {
                case MOVE_W: next_pos.y -= 1; break;
                case MOVE_S: next_pos.y += 1; break;
                case MOVE_D: next_pos.x += 1; break;
                case MOVE_A: next_pos.x -= 1; break;
                default: break;
            }

            if (i == 0 && next_pos == train_model.cell_pos)
                continue;

            if (i && next_pos == cargos[i - 1].cell_pos)
                continue;

            auto neighbors = get_valid_neighbors(terrain, next_pos, wagon.orientation);

            bool is_intersection = neighbors.size() >= 2;

            if (is_intersection) {
                if (!wagon.train_movements.empty()) {
                    wagon.next_move = wagon.train_movements.front();
                    wagon.train_movements.pop();
                } else {
                    wagon.next_move = NONE;
                }

                if (wagon.next_move != NONE &&
                    std::find(neighbors.begin(), neighbors.end(), wagon.next_move) != neighbors.end()) {
                    wagon.orientation = wagon.next_move;

                    wagon.next_move = NONE;
                } else {
                    wagon.progress = 1.0f;
                    continue;
                }
            } else {
                wagon.orientation = neighbors[0];
            }

            wagon.cell_pos = next_pos;
            wagon.progress -= 1.0f;
        } else{
            wagon.progress += (TRAIN_SPEED * delta_time) / cell_distance;
        }
    }
}

void Train::on_key_press(int key, int mods) {
    if (train_model.orientation == NONE) return;

    MOVE current = train_model.orientation;

    // forward/backward
    if (key == GLFW_KEY_W) {
        train_model.next_move = current; // move forward
        button_pressed = MOVE_W;
    }

    // left/right relative to current orientation
    if (key == GLFW_KEY_A) {
        switch (current) {
            case MOVE_W: train_model.next_move = MOVE_A; break; // left of north = west
            case MOVE_S: train_model.next_move = MOVE_D; break; // left of south = east
            case MOVE_A: train_model.next_move = MOVE_S; break; // left of west = south
            case MOVE_D: train_model.next_move = MOVE_W; break; // left of east = north
            default: break;
        }

        button_pressed = MOVE_A;
    }

    if (key == GLFW_KEY_D) {
        switch (current) {
            case MOVE_W: train_model.next_move = MOVE_D; break; // right of north = east
            case MOVE_S: train_model.next_move = MOVE_A; break; // right of south = west
            case MOVE_A: train_model.next_move = MOVE_W; break; // right of west = north
            case MOVE_D: train_model.next_move = MOVE_S; break; // right of east = south
            default: break;
        }

        button_pressed = MOVE_D;
    }
}

void Train::on_mouse_scroll(int mouseX, int mouseY, int offsetX, int offsetY) {
    distance_to_camera -= 160.0f * offsetY * delta_time;
    distance_to_camera = glm::clamp(0.0f, 100.0f, distance_to_camera);
}

bool Train::is_under_terrain() {
    return in_tunnel;
}

glm::vec3 basic_model::get_pos(Terrain *terrain) {
    glm::mat4 model = get_model(terrain, 1.0f);
    return glm::vec3(model[3]);
}

MOVE Train::get_next_move() {
    return train_model.next_move;
}

MOVE Train::get_button_pressed() {
    return button_pressed;
}
