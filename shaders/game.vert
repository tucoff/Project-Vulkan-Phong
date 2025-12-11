#version 450

layout(binding = 0) uniform UniformBufferObject 
{
    mat4 model;
    mat4 view;
    mat4 proj;
    vec3 lightPos;
    vec3 viewPos;
    vec4 lightAmbient;
    vec4 lightDiffuse;
    vec4 lightSpecular;
    vec4 matAmbient;
    vec4 matDiffuse;
    vec4 matSpecular;
    float shininess;
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec3 fragPosView;
layout(location = 1) out vec3 fragNormalView;

void main() 
{ 
    vec4 posView = ubo.view * ubo.model * vec4(inPosition, 1.0);
    fragPosView = posView.xyz;
     
    fragNormalView = mat3(transpose(inverse(ubo.view * ubo.model))) * inNormal;
     
    gl_Position = ubo.proj * posView;
}


