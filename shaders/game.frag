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
     
    float dotLN = dot(L, N);
    float dotVN = dot(V, N);
    
    vec3 diffuse = vec3(0.0);
    vec3 specular = vec3(0.0);
     
    if (dotLN < 0.0)
    {
        if (dotVN < 0.0)
        { 
            N = -N;
            dotLN = dot(L, N);
        }
        else
        { 
            outColor = vec4(ambient, 1.0);
            return;
        }
    }
    else if (dotVN < 0.0)
    { 
        outColor = vec4(ambient, 1.0);
        return;
    }
     
    diffuse = ubo.lightDiffuse.rgb * (dotLN * ubo.matDiffuse.rgb);
     
    vec3 R = reflect(-L, N);
    float dotRV = dot(R, V);
    
    if (dotRV > 0.0)
    {
        float specFactor = pow(dotRV, ubo.shininess);
        specular = ubo.lightSpecular.rgb * (ubo.matSpecular.rgb * specFactor);
    }
     
    vec3 finalColor = ambient + diffuse + specular;
    finalColor = clamp(finalColor, vec3(0.0), vec3(1.0));
    
    outColor = vec4(finalColor, 1.0);
}

