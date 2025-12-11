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

layout(location = 0) in vec3 fragPosView;
layout(location = 1) in vec3 fragNormalView;

layout(location = 0) out vec4 outColor;

void main() 
{ 
    vec3 N = normalize(fragNormalView);
    vec3 V = normalize(-fragPosView);
    vec3 lightPosView = (ubo.view * vec4(ubo.lightPos, 1.0)).xyz;
    vec3 L = normalize(lightPosView - fragPosView);  
 
    vec3 ambient = ubo.lightAmbient.rgb * ubo.matAmbient.rgb;
 
    float diff = max(dot(L, N), 0.0);
    vec3 diffuse = ubo.lightDiffuse.rgb * (diff * ubo.matDiffuse.rgb);
 
    vec3 specular = vec3(0.0);
    if (diff > 0.0) {
        vec3 H = normalize(L + V); 
        float spec = pow(max(dot(N, H), 0.0), ubo.shininess);
        specular = ubo.lightSpecular.rgb * (ubo.matSpecular.rgb * spec);
    }

    vec3 finalColor = ambient + diffuse + specular;
    outColor = vec4(finalColor, 1.0);
}

