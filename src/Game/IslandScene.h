#pragma once

#include "components/text_renderer.h"
#include "components/simple_scene.h"
#include "Camera/Camera.h"
#include "Terrain.h"
#include "trains.h"

namespace m1
{
    class IslandScene : public gfxc::SimpleScene
    {
     public:
        IslandScene();
        ~IslandScene();

        void Init() override;

        Mesh *CreateMesh(const char *name, const std::vector<VertexFormat> &vertices, const std::vector<unsigned int> &indices);

     private:
        void FrameStart() override;

        void draw_scene(float deltaTimeSeconds, int minimap = 0);
        void draw_ui(float deltaTimeSeconds, glm::vec3 highlight_color);

        void Update(float deltaTimeSeconds) override;
        void FrameEnd() override;

        void RenderUI(Mesh *mesh, Shader *shader, const glm::mat4 &modelMatrix, int highlighted);
        void MyRenderMesh(Mesh *mesh, Shader *shader, const glm::mat4 &modelMatrix, float alpha, int minimap = 0);

        void OnInputUpdate(float deltaTime, int mods) override;
        void OnKeyPress(int key, int mods) override;
        void OnKeyRelease(int key, int mods) override;
        void OnMouseMove(int mouseX, int mouseY, int deltaX, int deltaY) override;
        void OnMouseBtnPress(int mouseX, int mouseY, int button, int mods) override;
        void OnMouseBtnRelease(int mouseX, int mouseY, int button, int mods) override;
        void OnMouseScroll(int mouseX, int mouseY, int offsetX, int offsetY) override;
        void OnWindowResize(int width, int height) override;

     protected:
        implemented::Camera *camera;
        glm::mat4 projectionMatrix;
        glm::mat4 viewMatrix;
        bool renderCameraTarget;

        float fov;
        float near;
        float far;

        struct light_source {
            int  type;
            glm::vec3 position;
            glm::vec3 color;
            glm::vec3 direction;
        };

        light_source light_sources[1] = {
            {2, glm::vec3(0.0f, 10.0f, 0.0f), glm::vec3(1.0f), glm::normalize(glm::vec3(-0.3f, -1.0f, -0.2f))}
        };

        float elapsed_time;
        bool right_click_press;

        Terrain *terrain;
        Train *train;

        gfxc::TextRenderer *text;
    };
}   // namespace m1
