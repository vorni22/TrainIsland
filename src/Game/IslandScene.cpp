#include "IslandScene.h"

#include <vector>
#include <string>
#include <iostream>
#include "trains.h"

using namespace std;
using namespace m1;


/*
 *  To find out more about `FrameStart`, `Update`, `FrameEnd`
 *  and the order in which they are called, see `world.cpp`.
 */


IslandScene::IslandScene()
{
}


IslandScene::~IslandScene()
{
}


void IslandScene::Init()
{
    renderCameraTarget = false;
    right_click_press = false;

    camera = new implemented::Camera();
    camera->Set(glm::vec3(0, 2, 3.5f), glm::vec3(0, 1, 0), glm::vec3(0, 1, 0));

    Shader *shader = new Shader("MainShader");
    shader->AddShader(PATH_JOIN(window->props.selfDir, "src", "Game", "shaders", "VertexShader.glsl"), GL_VERTEX_SHADER);
    shader->AddShader(PATH_JOIN(window->props.selfDir, "src", "Game", "shaders", "FragmentShader.glsl"), GL_FRAGMENT_SHADER);
    shader->CreateAndLink();
    shaders[shader->GetName()] = shader;

    Shader *OccludedShader = new Shader("OccludedShader");
    OccludedShader->AddShader(PATH_JOIN(window->props.selfDir, "src", "Game", "shaders", "VertexOccluded.glsl"), GL_VERTEX_SHADER);
    OccludedShader->AddShader(PATH_JOIN(window->props.selfDir, "src", "Game", "shaders", "FragmentOccluded.glsl"), GL_FRAGMENT_SHADER);
    OccludedShader->CreateAndLink();
    shaders[OccludedShader->GetName()] = OccludedShader;

    Shader *UIShader = new Shader("UIShader");
    UIShader->AddShader(PATH_JOIN(window->props.selfDir, "src", "Game", "shaders", "VertexUI.glsl"), GL_VERTEX_SHADER);
    UIShader->AddShader(PATH_JOIN(window->props.selfDir, "src", "Game", "shaders", "FragmentUI.glsl"), GL_FRAGMENT_SHADER);
    UIShader->CreateAndLink();
    shaders[UIShader->GetName()] = UIShader;

    text = new gfxc::TextRenderer(window->props.selfDir, window->GetResolution().x, window->GetResolution().y);
    text->Load(PATH_JOIN(window->props.selfDir, RESOURCE_PATH::FONTS, "Hack-Bold.ttf"), 72);

    fov = 60.0f;
    near = 0.1f;
    far = 700.0f;

    int terrain_len = 648;

    {
        std::vector<unsigned int> indices;
        std::vector<VertexFormat> vertices;

        terrain = new Terrain(terrain_len, terrain_len, [this](const char *name, const std::vector<VertexFormat> &vertices, const std::vector<unsigned int> &indices) {
            this->CreateMesh(name, vertices, indices);
        });
        terrain->generate_mesh(vertices, indices);
        Mesh* terrain_mesh = CreateMesh("terrain", vertices, indices);

        indices.clear();
        vertices.clear();
        terrain->generate_rail_mesh(vertices, indices);
        Mesh *rails = CreateMesh("rails", vertices, indices);

        indices.clear();
        vertices.clear();
        terrain->generate_under_terrain_mesh(vertices, indices);
        Mesh *under = CreateMesh("under", vertices, indices);

        train = new Train(terrain, 3, GRID_CELL_SIZE, [this](const char *name, const std::vector<VertexFormat> &vertices, const std::vector<unsigned int> &indices) {
            this->CreateMesh(name, vertices, indices);
        });
        
    }

    {   
        float h = (5.0f) - 1.0f;
        glm::vec3 A = glm::vec3(-terrain_len, h, -terrain_len);
        glm::vec3 B = glm::vec3(-terrain_len, h, 2.0f * terrain_len);
        glm::vec3 C = glm::vec3(2.0f * terrain_len, h, 2.0f * terrain_len);
        glm::vec3 D = glm::vec3(2.0f * terrain_len, h, -terrain_len);
        glm::vec3 color = glm::vec3(0.008, 0.045, 1);

        std::vector<VertexFormat> vertices = {
            VertexFormat(A, color, glm::vec3(0, 1, 0)),
            VertexFormat(B, color, glm::vec3(0, 1, 0)),
            VertexFormat(C, color, glm::vec3(0, 1, 0)),
            VertexFormat(D, color, glm::vec3(0, 1, 0))
        };
        std::vector<unsigned int> indices = {
            0, 1, 3,
            1, 2, 3
        };

        Mesh* orangeCube = CreateMesh("sea_level", vertices, indices);

        color = glm::vec3(0.94, 0.87, 0.65);
    }

    {
        float h = 0.0f;
        glm::vec3 A = glm::vec3(-terrain_len, h, -terrain_len);
        glm::vec3 B = glm::vec3(-terrain_len, h, 2.0f * terrain_len);
        glm::vec3 C = glm::vec3(2.0f * terrain_len, h, 2.0f * terrain_len);
        glm::vec3 D = glm::vec3(2.0f * terrain_len, h, -terrain_len);
        glm::vec3 color = glm::vec3(0.94, 0.87, 0.65);

        std::vector<VertexFormat> vertices = {
            VertexFormat(A, color, glm::vec3(0, 1, 0)),
            VertexFormat(B, color, glm::vec3(0, 1, 0)),
            VertexFormat(C, color, glm::vec3(0, 1, 0)),
            VertexFormat(D, color, glm::vec3(0, 1, 0))
        };
        std::vector<unsigned int> indices = {
            0, 1, 3,
            1, 2, 3
        };

        Mesh* orangeCube = CreateMesh("extension", vertices, indices);
    }

    {
        std::vector<VertexFormat> v = {
            VertexFormat(glm::vec3(0, 20, 0), glm::vec3(1, 1, 1)),
            VertexFormat(glm::vec3(-15, -10, 0), glm::vec3(1, 1, 1)),
            VertexFormat(glm::vec3(15, -10, 0), glm::vec3(1, 1, 1)),
        };

        std::vector<unsigned int> ind = {0, 1, 2};

        CreateMesh("ui_triangle", v, ind);
    }

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    elapsed_time = 0.0f;
}


void IslandScene::FrameStart()
{
    // Clears the color buffer (using the previously set color) and depth buffer
    glClearColor(0.604, 0.988, 0.98, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glm::ivec2 resolution = window->GetResolution();
    // Sets the screen area where to draw
    glViewport(0, 0, resolution.x, resolution.y);
}

void IslandScene::Update(float deltaTimeSeconds)
{   
    train->update(deltaTimeSeconds, camera);
    terrain->update_stations(deltaTimeSeconds);
    elapsed_time += deltaTimeSeconds;
    
    {
        projectionMatrix = glm::perspective(RADIANS(fov), window->props.aspectRatio, near, far);
        viewMatrix = camera->GetViewMatrix();

        draw_scene(deltaTimeSeconds, 0);
    }

    int winW = window->GetResolution().x;
    int winH = window->GetResolution().y;

    // MINIMAP
    {
        int terrain_len = 648;
        int size = winH * 0.5f;

        glViewport(10, 10, size, size);
        float half = terrain_len / 2.0f;

        glm::mat4 minimapProj = glm::ortho(
            -half, +half,
            -half, +half,
            -1000.0f, 1000.0f
        );

        projectionMatrix = minimapProj;

        float H = 400;
        glm::vec3 eye    = glm::vec3(half, H, half);
        glm::vec3 target = glm::vec3(half, 0.0f, half);
        glm::vec3 up     = glm::vec3(0, 0, -1);

        viewMatrix = glm::lookAt(eye, target, up);

        glClear(GL_DEPTH_BUFFER_BIT);
        draw_scene(deltaTimeSeconds, 1);
    }

    // UI
    {
        glViewport(0, 0, winW, winH);

        glClear(GL_DEPTH_BUFFER_BIT);
        draw_ui(deltaTimeSeconds, glm::vec3(0.012, 1, 0));
    }
}

void IslandScene::FrameEnd()
{
    //DrawCoordinateSystem(camera->GetViewMatrix(), projectionMatrix);
}


void IslandScene::MyRenderMesh(Mesh * mesh, Shader * shader, const glm::mat4 & modelMatrix, float alpha, int minimap)
{
    if (!mesh || !shader || !shader->GetProgramID())
        return;

    // Render an object using the specified shader and the specified position
    glUseProgram(shader->program);

    for (int i = 0; i < 1; ++i)
    {
        std::string name_position = std::string("lights[") + std::to_string(i) + std::string("].position");
        GLuint location = glGetUniformLocation(shader->program, name_position.c_str());
        glUniform3fv(location, 1, glm::value_ptr(light_sources[i].position));

        std::string name_color = std::string("lights[") + std::to_string(i) + std::string("].color");
        location = glGetUniformLocation(shader->program, name_color.c_str());
        glUniform3fv(location, 1, glm::value_ptr(light_sources[i].color));

        std::string name_type = std::string("lights[") + std::to_string(i) + std::string("].type");
        location = glGetUniformLocation(shader->program, name_type.c_str());
        glUniform1i(location, light_sources[i].type);

        std::string name_direction = std::string("lights[") + std::to_string(i) + std::string("].direction");
        location = glGetUniformLocation(shader->program, name_direction.c_str());
        glUniform3fv(location, 1, glm::value_ptr(light_sources[i].direction));
    }

    GLuint minimap_loc = glGetUniformLocation(shader->program, "minimap");
    glUniform1i(minimap_loc, minimap);

    int cutoff_angle_loc = glGetUniformLocation(shader->program, "cutoff_angle");
    glUniform1f(cutoff_angle_loc, 60.0f);

    // Set eye position (camera position) uniform
    glm::vec3 eyePosition = camera->position;
    int eye_position = glGetUniformLocation(shader->program, "eye_position");
    glUniform3f(eye_position, eyePosition.x, eyePosition.y, eyePosition.z);

    // Set material property uniforms (shininess, kd, ks, object color) 
    int material_shininess = glGetUniformLocation(shader->program, "material_shininess");
    glUniform1i(material_shininess, 4);

    int material_kd = glGetUniformLocation(shader->program, "material_kd");
    glUniform1f(material_kd, 1.0f);

    int material_ks = glGetUniformLocation(shader->program, "material_ks");
    glUniform1f(material_ks, 0.7f);

    int alpha_loc = glGetUniformLocation(shader->program, "alpha");
    glUniform1f(alpha_loc, alpha);

    // Bind model matrix
    GLint loc_model_matrix = glGetUniformLocation(shader->program, "Model");
    glUniformMatrix4fv(loc_model_matrix, 1, GL_FALSE, glm::value_ptr(modelMatrix));

    // Bind view matrix
    int loc_view_matrix = glGetUniformLocation(shader->program, "View");
    glUniformMatrix4fv(loc_view_matrix, 1, GL_FALSE, glm::value_ptr(viewMatrix));

    // Bind projection matrix
    int loc_projection_matrix = glGetUniformLocation(shader->program, "Projection");
    glUniformMatrix4fv(loc_projection_matrix, 1, GL_FALSE, glm::value_ptr(projectionMatrix));

    // Draw the object
    glBindVertexArray(mesh->GetBuffers()->m_VAO);
    glDrawElements(mesh->GetDrawMode(), static_cast<int>(mesh->indices.size()), GL_UNSIGNED_INT, 0);
}


/*
 *  These are callback functions. To find more about callbacks and
 *  how they behave, see `input_controller.h`.
 */


void IslandScene::OnInputUpdate(float deltaTime, int mods)
{
}


void IslandScene::OnKeyPress(int key, int mods)
{
    train->on_key_press(key, mods);

    // Add key press event
    if (key == GLFW_KEY_T)
    {
        renderCameraTarget = !renderCameraTarget;
    }
}


void IslandScene::OnKeyRelease(int key, int mods)
{
    // Add key release event
}


void IslandScene::OnMouseMove(int mouseX, int mouseY, int deltaX, int deltaY)
{
    // Add mouse move event

    if (window->MouseHold(GLFW_MOUSE_BUTTON_RIGHT))
    {
        float sensivityOX = 0.001f;
        float sensivityOY = 0.001f;

        if (window->GetSpecialKeyState() == 0) {
            renderCameraTarget = false;
            // TODO(student): Rotate the camera in first-person mode around
            // OX and OY using `deltaX` and `deltaY`. Use the sensitivity
            // variables for setting up the rotation speed.

            //camera->RotateFirstPerson_OY(-deltaX * sensivityOX);
            //camera->RotateFirstPerson_OX(-deltaY * sensivityOY);
            camera->RotateThirdPerson_OY(-deltaX * sensivityOX);
            camera->RotateThirdPerson_OX(-deltaY * sensivityOY);
        }

        if (window->GetSpecialKeyState() & GLFW_MOD_CONTROL) {
            renderCameraTarget = true;
            // TODO(student): Rotate the camera in third-person mode around
            // OX and OY using `deltaX` and `deltaY`. Use the sensitivity
            // variables for setting up the rotation speed.

            //camera->RotateThirdPerson_OY(-deltaX * sensivityOX);
            //camera->RotateThirdPerson_OX(-deltaY * sensivityOY);
            camera->RotateThirdPerson_OY(-deltaX * sensivityOX);
            camera->RotateThirdPerson_OX(-deltaY * sensivityOY);
        }
    }
}

Mesh* IslandScene::CreateMesh(const char *name, const std::vector<VertexFormat> &vertices, const std::vector<unsigned int> &indices)
{
    unsigned int VAO = 0;
    // Create the VAO and bind it
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    // Create the VBO and bind it
    unsigned int VBO;
    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);

    // Send vertices data into the VBO buffer
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices[0]) * vertices.size(), &vertices[0], GL_STATIC_DRAW);

    // Create the IBO and bind it
    unsigned int IBO;
    glGenBuffers(1, &IBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IBO);

    // Send indices data into the IBO buffer
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices[0]) * indices.size(), &indices[0], GL_STATIC_DRAW);

    // ========================================================================

    // Set vertex position attribute
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexFormat), 0);

    // Set vertex normal attribute
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexFormat), (void*)(sizeof(glm::vec3)));

    // Set texture coordinate attribute
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexFormat), (void*)(2 * sizeof(glm::vec3)));

    // Set vertex color attribute
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(VertexFormat), (void*)(2 * sizeof(glm::vec3) + sizeof(glm::vec2)));
    // ========================================================================

    // Unbind the VAO
    glBindVertexArray(0);

    // Check for OpenGL errors
    CheckOpenGLError();

    // Mesh information is saved into a Mesh object
    meshes[name] = new Mesh(name);
    meshes[name]->InitFromBuffer(VAO, static_cast<unsigned int>(indices.size()));
    meshes[name]->vertices = vertices;
    meshes[name]->indices = indices;
    return meshes[name];
}


void IslandScene::OnMouseBtnPress(int mouseX, int mouseY, int button, int mods)
{
    // Add mouse button press event
}


void IslandScene::OnMouseBtnRelease(int mouseX, int mouseY, int button, int mods)
{
    // Add mouse button release event
}


void IslandScene::OnMouseScroll(int mouseX, int mouseY, int offsetX, int offsetY)
{
    train->on_mouse_scroll(mouseX, mouseY, offsetX, offsetY);
}


void IslandScene::OnWindowResize(int width, int height)
{
    delete text;
    text = new gfxc::TextRenderer(window->props.selfDir, window->GetResolution().x, window->GetResolution().y);
    text->Load(PATH_JOIN(window->props.selfDir, RESOURCE_PATH::FONTS, "Hack-Bold.ttf"), 72);
}

void m1::IslandScene::draw_scene(float deltaTimeSeconds, int minimap) {
    glClear(GL_DEPTH_BUFFER_BIT);

    // Draw terrain and rails first
    if (!minimap) {
        glm::mat4 modelMatrix = glm::mat4(1);
        glDisable(GL_DEPTH_TEST);
        MyRenderMesh(meshes["extension"], shaders["MainShader"], modelMatrix, 1.0f, minimap);
        glEnable(GL_DEPTH_TEST);
    }

    {
        glm::mat4 modelMatrix = glm::mat4(1);
        modelMatrix = glm::scale(modelMatrix, glm::vec3(1.0f));
        MyRenderMesh(meshes["terrain"], shaders["MainShader"], modelMatrix, 1.0f, minimap);

        terrain->draw([this, minimap](const char *name, const glm::mat4& model, float alpha) {
                MyRenderMesh(meshes[name], shaders["MainShader"], model, alpha, minimap);
        });

        if (minimap) {
            glDisable(GL_DEPTH_TEST);
            MyRenderMesh(meshes["rails"], shaders["MainShader"], modelMatrix, 1.0f, minimap);
            glEnable(GL_DEPTH_TEST);
        } else MyRenderMesh(meshes["rails"], shaders["MainShader"], modelMatrix, 1.0f, minimap);
    }

    {
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glm::mat4 modelMatrix = glm::mat4(1);
        MyRenderMesh(meshes["sea_level"], shaders["MainShader"], modelMatrix, 0.5f, minimap);
    }

    if (train->is_under_terrain() && !minimap) {
        glDisable(GL_DEPTH_TEST);
        glm::mat4 modelMatrix = glm::mat4(1);
        modelMatrix = glm::scale(modelMatrix, glm::vec3(1.0f));
        MyRenderMesh(meshes["under"], shaders["MainShader"], modelMatrix, 0.8f, minimap);
        glEnable(GL_DEPTH_TEST);
    }
    
    if (!minimap) {
        glDisable(GL_CULL_FACE);
        glDepthFunc(GL_GREATER); // Only draw where depth test FAILS (behind objects)
        train->draw_outline([this, minimap](const char *name, const glm::mat4& model, float alpha) {
            MyRenderMesh(meshes[name], shaders["OccludedShader"], model, alpha, minimap);
        }, 1.1f);
        glDepthFunc(GL_LESS); // Restore normal depth testing
        glEnable(GL_CULL_FACE);
    }
    
    {
        if (!minimap) {
            train->draw([this, minimap](const char *name, const glm::mat4& model, float alpha) {
                MyRenderMesh(meshes[name], shaders["MainShader"], model, alpha, minimap);
            });
        } else {
            glDisable(GL_DEPTH_TEST);
            train->draw_outline([this, minimap](const char *name, const glm::mat4& model, float alpha) {
                MyRenderMesh(meshes[name], shaders["MainShader"], model, alpha, minimap);
            }, 2.0f);
            glEnable(GL_DEPTH_TEST);
        }
    }
}

void m1::IslandScene::draw_ui(float deltaTimeSeconds, glm::vec3 highlight_color) {
    glm::ivec2 res = window->GetResolution();
    float cx = res.x / 2.0f;
    float baseY = 60.0f;

    Shader* shader = shaders["UIShader"];
    glUseProgram(shader->program);

    glm::mat4 proj = glm::ortho(0.0f, (float)res.x,
                                0.0f, (float)res.y,
                                -1.0f, 1.0f);

    glUniformMatrix4fv(glGetUniformLocation(shader->program, "Projection"), 1, GL_FALSE, glm::value_ptr(proj));

    GLuint location = glGetUniformLocation(shader->program, "highlightColor");
    glUniform3fv(location, 1, glm::value_ptr(highlight_color));

    {
        glm::mat4 model = glm::translate(glm::mat4(1), glm::vec3(cx, baseY + 40, 0));

        int highlighted = 0;
        if (train->get_button_pressed() == MOVE_W) {
            model = glm::scale(model, glm::vec3(1.3f));
            highlighted = 1;
        }

        RenderUI(meshes["ui_triangle"], shader, model, highlighted);
    }

    // --- A triangle (rotated left) ---
    {
        glm::mat4 model = glm::translate(glm::mat4(1), glm::vec3(cx - 50, baseY, 0));
        model = glm::rotate(model, glm::radians(90.0f), glm::vec3(0,0,1));

        int highlighted = 0;
        if (train->get_button_pressed() == MOVE_A) {
            model = glm::scale(model, glm::vec3(1.3f));
            highlighted = 1;
        }

        RenderUI(meshes["ui_triangle"], shader, model, highlighted);
    }

    // --- D triangle (rotated right) ---
    {
        glm::mat4 model = glm::translate(glm::mat4(1), glm::vec3(cx + 50, baseY, 0));
        model = glm::rotate(model, glm::radians(-90.0f), glm::vec3(0,0,1));

        int highlighted = 0;
        if (train->get_button_pressed() == MOVE_D) {
            model = glm::scale(model, glm::vec3(1.3f));
            highlighted = 1;
        }

        RenderUI(meshes["ui_triangle"], shader, model, highlighted);
    }

    float mission_time = terrain->get_mission_time();
    text->RenderText("Time: " + std::to_string(int(mission_time)) + "/120", 10, 10, 0.25f, glm::vec3(1.0f));

    if (terrain->request_size() == 0) {
        text->RenderText("Go back to the main station!", 10, 40, 0.25f, glm::vec3(1.0f));
    }

    terrain->draw_ui([this](const char *name, const glm::mat4& model, float alpha) {
        MyRenderMesh(meshes[name], shaders["MainShader"], model, alpha);
    }, res.x, res.y);

    if (terrain->is_gameover()) {
        text->RenderText("GAME OVER", res.x / 2.0f - 200, res.y / 2.0f, 1.0f, glm::vec3(1.0f));
    }
}

void m1::IslandScene::RenderUI(Mesh *mesh, Shader *shader, const glm::mat4 &modelMatrix, int highlighted) {
    if (!mesh || !shader || !shader->GetProgramID())
        return;

    glUniformMatrix4fv(glGetUniformLocation(shader->program, "Model"), 1, GL_FALSE, glm::value_ptr(modelMatrix));

    GLuint is_highlighted_location = glGetUniformLocation(shader->program, "isHighlighted");
    glUniform1i(is_highlighted_location, highlighted);

    glBindVertexArray(mesh->GetBuffers()->m_VAO);
    glDrawElements(mesh->GetDrawMode(), static_cast<int>(mesh->indices.size()), GL_UNSIGNED_INT, 0);
}
