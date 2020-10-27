precision lowp float;
precision lowp sampler2D;

varying vec3 cameraNormal;
varying vec3 cameraLightDirection;
varying vec2 texCoord;

uniform sampler2D tex;

void main()
{
	vec4 materialColor = texture2D(tex, texCoord);

    vec3 normal = normalize(cameraNormal);
    vec3 lightDirection = normalize(cameraLightDirection);
    float cosTheta = clamp(dot(normal, lightDirection), 0.0, 1.0);

    vec3 lightAmbientColor = vec3(0.3, 0.3, 0.3);
    vec3 lightDiffuseColor = vec3(1.0, 1.0, 1.0);

    gl_FragColor =
        materialColor * vec4(lightAmbientColor, 1) +
        materialColor * vec4(lightDiffuseColor, 1) * cosTheta;
}