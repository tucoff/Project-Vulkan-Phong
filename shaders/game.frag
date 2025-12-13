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
    vec3 L = normalize(ubo.lightPos - fragPosView);  
 
    // Alfa do material ambiente define a opacidade base (sólida)
    float baseOpacity = ubo.matAmbient.a;
    
    // Alfas de diffuse/specular modulam apenas a intensidade de seus termos
    float intensityAmbient = ubo.lightAmbient.a;
    float intensityDiffuse = ubo.lightDiffuse.a * ubo.matDiffuse.a;
    float intensitySpecular = ubo.lightSpecular.a * ubo.matSpecular.a;

    vec3 ambient = (ubo.lightAmbient.rgb * ubo.matAmbient.rgb) * intensityAmbient;
 
    float diff = max(dot(L, N), 0.0);
    vec3 diffuse = (ubo.lightDiffuse.rgb * (diff * ubo.matDiffuse.rgb)) * intensityDiffuse;
 
    vec3 specular = vec3(0.0);
    if (diff > 0.0) {
        vec3 R = reflect(-L, N);
        float spec = pow(max(dot(R, V), 0.0), ubo.shininess);
        specular = (ubo.lightSpecular.rgb * (ubo.matSpecular.rgb * spec)) * intensitySpecular;
    }

    vec3 finalColor = ambient + diffuse + specular;

    // Opacidade final = alfa ambiente (não depende de diffuse/specular)
    outColor = vec4(finalColor, baseOpacity);
}

