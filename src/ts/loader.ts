import { vec2, vec3 } from 'gl-matrix';

class ParsedObjectFile {
    vertices: Array<vec3> = [];
    texCoords: Array<vec2> = [];
    normals: Array<vec3> = [];

    faces: Array<Array<Array<number>>> = [];
}

export class ObjFile {
    buffer: WebGLBuffer;
    triangleCount: number;
}

class ParsedTetFile {
    // all vertices
    vertices: Array<number> = [];
    // all texCoords
    texCoords: Array<number> = [];

    // indices [v, t, v, t v, t] per face
    faces: Array<number> = [];
    // indices [v, v, v, v] per tet
    tetsIndices: Array<number> = [];
}

export class TetFile {
    vertices: Float32Array;
    tetsIndices: Uint32Array;

    // list of vertices that we actually need to transfer from simulation back to render
    visibleIndices: Uint32Array;

    visibleTexCoords: Float32Array;

    // we need this to recreate visible buffers after simulation have returned the vertices
    faces: Uint32Array;

    // this is recalculated each time we get updated by simlation
    visibleVertices: Float32Array;
    // this is recalculated also
    visibleNormals: Float32Array;

    vertexBuffer: WebGLBuffer;
    texCoordBuffer: WebGLBuffer;
    normalBuffer: WebGLBuffer;

    triangleCount: number;

    public updateVisibleVertices(vertices: Float32Array): void {
        for (let i = 0; i < this.visibleIndices.length; i++) {
            const index = this.visibleIndices[i];

            this.vertices[index * 3 + 0] = vertices[i * 3 + 0];
            this.vertices[index * 3 + 1] = vertices[i * 3 + 1];
            this.vertices[index * 3 + 2] = vertices[i * 3 + 2];
        }

        this.recalcVisibleVertices();
        this.recalcNormals();
    }

    public recalcVisibleVertices(): void {
        let j = 0;
        for (let i = 0; i < this.faces.length; i += 2) {
            const vertexIndex = this.faces[i + 0];

            this.visibleVertices[j + 0] = this.vertices[vertexIndex * 3 + 0];
            this.visibleVertices[j + 1] = this.vertices[vertexIndex * 3 + 1];
            this.visibleVertices[j + 2] = this.vertices[vertexIndex * 3 + 2];
            j += 3;
        }
    }

    public recalcNormals(): void {
        // TODO
    }
}

export class CompiledShader {
    program: WebGLProgram;
}

export class Loader {
    gl: WebGLRenderingContext;

    constructor(gl: WebGLRenderingContext) {
        this.gl = gl;
    }

    async fetchText(url: string): Promise<string> {
        return new Promise<string>(async (resolve, reject) => {
            const response = await fetch(url);
            if (response.status !== 200) {
                reject('Failed to fetch file.');
                return;
            }

            resolve(await response.text());
        });
    }

    async loadObj(url: string, size: number): Promise<ObjFile> {
        const gl = this.gl;
        return new Promise<ObjFile>(async (resolve, reject) => {
            const obj: ParsedObjectFile = this.parseObjFile(await this.fetchText(url));

            const data = new Float32Array(obj.vertices.length * (3 + 3 + 2));
            let j = 0;
            for (let i = 0; i < obj.faces.length; i++) {
                const face = obj.faces[i];

                for (let k = 0; k < face.length; k++) {
                    const v = face[k];

                    const vertex = obj.vertices[v[0]];
                    const texCoord = obj.texCoords[v[1]];
                    const normal = obj.normals[v[2]];

                    data[j++] = vertex[0] * size;
                    data[j++] = vertex[1] * size;
                    data[j++] = vertex[2] * size;

                    data[j++] = texCoord[0];
                    data[j++] = texCoord[1];

                    data[j++] = normal[0];
                    data[j++] = normal[1];
                    data[j++] = normal[2];
                }
            }

            const buffer = gl.createBuffer();
            gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
            gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);

            const result = new ObjFile();
            result.buffer = buffer;
            result.triangleCount = obj.faces.length;

            resolve(result);
        });
    }

    parseObjFile(text: string): ParsedObjectFile {
        const result = new ParsedObjectFile();
        const lines: Array<string> = text.split(/\r?\n/);
        lines.forEach((line: string): void => {
            line = line.trim();
            if (line.startsWith('#')) return;
            const values: Array<string> = line.split(' ');
            if (values.length < 1) return;
            switch (values[0]) {
                case 'v': {
                    const v = vec3.fromValues(parseFloat(values[1]), parseFloat(values[2]), parseFloat(values[3]));
                    result.vertices.push(v);
                    break;
                }
                case 'vt': {
                    const t = vec2.fromValues(parseFloat(values[1]), parseFloat(values[2]));
                    result.texCoords.push(t);
                    break;
                }
                case 'vn': {
                    const n = vec3.fromValues(parseFloat(values[1]), parseFloat(values[2]), parseFloat(values[3]));
                    result.normals.push(n);
                    break;
                }
                case 'f': {
                    const face: Array<Array<number>> = [];
                    for (let i = 0; i < 3; i++) {
                        const indicesStrs: Array<string> = values[1 + i].split('/');
                        const indices: Array<number> = [];
                        for (let j = 0; j < indicesStrs.length; j++) {
                            indices.push(parseInt(indicesStrs[j]) - 1);
                        }
                        face.push(indices);
                    }
                    result.faces.push(face);
                    break;
                }
            }
        });
        return result;
    }

    async loadTet(url: string): Promise<TetFile> {
        const gl = this.gl;
        return new Promise<TetFile>(async (resolve, reject) => {
            const tet: ParsedTetFile = this.parseTetFile(await this.fetchText(url));

            const result = new TetFile();

            result.vertexBuffer = gl.createBuffer();
            result.triangleCount = tet.faces.length / 2;

            result.vertices = new Float32Array(tet.vertices);
            result.tetsIndices = new Uint32Array(tet.tetsIndices);
            result.faces = new Uint32Array(tet.faces);

            const visibleIndices: Array<number> = [];
            for (let i = 0; i < result.faces.length; i += 2) {
                const vertexIndex = result.faces[i];
                if (!visibleIndices.includes(vertexIndex)) {
                    visibleIndices.push(vertexIndex);
                }
            }
            result.visibleIndices = new Uint32Array(visibleIndices);

            result.visibleTexCoords = new Float32Array(result.triangleCount * 2);
            let j = 0;
            for (let i = 0; i < result.faces.length; i += 2) {
                const texCoordIndex = result.faces[i + 1];

                const u = tet.texCoords[texCoordIndex * 2 + 0];
                const v = tet.texCoords[texCoordIndex * 2 + 1];

                result.visibleTexCoords[j + 0] = u;
                result.visibleTexCoords[j + 1] = v;
                j += 2;
            }

            result.visibleVertices = new Float32Array(result.triangleCount * 3);
            result.visibleNormals = new Float32Array(result.triangleCount * 3);

            resolve(result);
        });
    }

    parseTetFile(text: string): ParsedTetFile {
        const result = new ParsedTetFile();
        const lines: Array<string> = text.split(/\r?\n/);
        lines.forEach((line: string): void => {
            line = line.trim();
            if (line.startsWith('#')) return;
            const values: Array<string> = line.split(' ');
            if (values.length < 1) return;
            switch (values[0]) {
                case 'v': {
                    result.vertices.push(parseFloat(values[1]));
                    result.vertices.push(parseFloat(values[2]));
                    result.vertices.push(parseFloat(values[3]));
                    break;
                }
                case 'vt': {
                    result.texCoords.push(parseFloat(values[1]));
                    result.texCoords.push(parseFloat(values[2]));
                    break;
                }
                case 'f': {
                    for (let i = 1; i < values.length; i++) {
                        const indicesStrs: Array<string> = values[i].split('/');
                        for (let j = 0; j < indicesStrs.length; j++) {
                            result.faces.push(parseInt(indicesStrs[j]) - 1);
                        }
                    }
                    break;
                }
                case 't': {
                    result.tetsIndices.push(parseInt(values[1]) - 1);
                    result.tetsIndices.push(parseInt(values[2]) - 1);
                    result.tetsIndices.push(parseInt(values[3]) - 1);
                    result.tetsIndices.push(parseInt(values[4]) - 1);
                    break;
                }
            }
        });
        return result;
    }

    async loadTexture(url: string): Promise<WebGLTexture> {
        const gl = this.gl;
        return new Promise<WebGLTexture>((resolve, reject) => {
            const tex = gl.createTexture();
            const image = new Image();
            image.onload = (): void => {
                gl.bindTexture(gl.TEXTURE_2D, tex);
                gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);

                /*
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
                */

                resolve(tex);
            };
            image.onerror = (): void => {
                reject('Failed to load an image.');
            };
            image.src = url;
        });
    }

    async loadShader(baseUrl: string, parameters: Array<string>): Promise<CompiledShader> {
        const gl = this.gl;
        return new Promise<CompiledShader>(async (resolve, reject) => {
            const vertText = await this.fetchText(`${baseUrl}.vert`);
            const fragText = await this.fetchText(`${baseUrl}.frag`);

            const vertexShader = gl.createShader(gl.VERTEX_SHADER);
            const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);

            gl.shaderSource(vertexShader, vertText);
            gl.shaderSource(fragmentShader, fragText);

            gl.compileShader(vertexShader);
            if (!gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS)) {
                reject(`ERROR compiling vertex shader for ${name}! ${gl.getShaderInfoLog(vertexShader)}`);
                return;
            }

            gl.compileShader(fragmentShader);
            if (!gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS)) {
                reject(`ERROR compiling fragment shader for ${name}! ${gl.getShaderInfoLog(fragmentShader)}`);
                return;
            }

            const program = gl.createProgram();
            gl.attachShader(program, vertexShader);
            gl.attachShader(program, fragmentShader);
            gl.linkProgram(program);
            if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
                reject(`ERROR linking program! ${gl.getProgramInfoLog(program)}`);
                return;
            }
            gl.validateProgram(program);
            if (!gl.getProgramParameter(program, gl.VALIDATE_STATUS)) {
                reject(`ERROR validating program! ${gl.getProgramInfoLog(program)}`);
                return;
            }

            const result = new CompiledShader();
            result.program = program;

            parameters.forEach((parameter: string): void => {
                const uniformLocation = gl.getUniformLocation(program, parameter);
                if (uniformLocation !== null) {
                    (result as any)[parameter] = uniformLocation;
                } else {
                    const attributeLocation = gl.getAttribLocation(program, parameter);
                    if (attributeLocation !== null) {
                        (result as any)[parameter] = attributeLocation;
                    } else {
                        console.warn(`${parameter} is not found in shader.`);
                    }
                }
            });

            resolve(result);
        });
    }
}
