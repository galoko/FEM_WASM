precision lowp float;
precision lowp sampler2D;

attribute vec3 vertexPosition;
attribute vec3 vertexNormal;
attribute vec2 vertexTexCoord;

varying vec3 cameraNormal;
varying vec3 cameraLightDirection;
varying vec2 texCoord;

uniform mat4 projection;
uniform mat4 view;

void main()
{
    gl_Position = projection * view * vec4(vertexPosition, 1);

    cameraLightDirection = -(view * vec4(vertexPosition, 1)).xyz;
    cameraNormal = (view * vec4(vertexNormal, 0)).xyz;
	
	texCoord = vertexTexCoord;
}