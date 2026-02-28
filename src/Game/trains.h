#pragma once

#include "Terrain.h"
#include "../Camera/Camera.h"
#include "components/simple_scene.h"
#include <vector>
#include <queue>
#include <functional>

enum MOVE {
    MOVE_S = 0,
    MOVE_D = 1,
    MOVE_W = 2,
    MOVE_A = 3,
    NONE = 4
};

struct basic_model {
    glm::ivec2 cell_pos;
    float progress;

    MOVE orientation;
    MOVE next_move;
    std::queue<MOVE> train_movements;

    glm::mat4 get_model(Terrain *terrain, float scale);
    glm::vec3 get_pos(Terrain *terrain);
};

class Train {
public:
    Train(Terrain *terrain, int cargo_count, float cell_size, create_mesh_func create_mesh);

    void draw(draw_mesh_func draw_mesh);
    void draw_outline(draw_mesh_func draw_mesh, float scale);

    void create_locomotive(std::vector<VertexFormat> &vertices, std::vector<uint32_t> &indices);
    void create_wagon(std::vector<VertexFormat> &vertices, std::vector<uint32_t> &indices);

    void init(glm::ivec2 start, create_mesh_func create_mesh);
    void update(float delta_time, implemented::Camera *camera);
    void update_wagons();

    void on_key_press(int key, int mods);
    void on_mouse_scroll(int mouseX, int mouseY, int offsetX, int offsetY);

    bool is_under_terrain();

    MOVE get_next_move();
    MOVE get_button_pressed();

private:

    void create_box(std::vector<VertexFormat> &vertices, std::vector<uint32_t> &indices,
                       uint32_t &currentIndex, float x, float y, float z,
                       float width, float height, float depth, glm::vec3 color);

    void create_cylinder(std::vector<VertexFormat> &vertices, std::vector<uint32_t> &indices,
                           uint32_t &currentIndex, float x, float y, float z,
                           float radius, float length, glm::vec3 color, int segments, char axis);

    glm::ivec2 train_pivot;
    basic_model train_model;
    std::vector<basic_model> cargos;

    MOVE button_pressed;

    Terrain *terrain;
    float cell;
    float distance_to_camera;
    float delta_time;

    int wagons;
    bool train_moved;
    bool in_tunnel;
};