import { vec2, vec3 } from 'gl-matrix';
import { sqrDist } from 'gl-matrix/src/gl-matrix/vec2';

class ParsedObjectFile {
    vertices: Array<vec3> = [];
    texCoords: Array<vec2> = [];
    normals: Array<vec3> = [];

    // [[v, t, n], [v, t, n], [v, t, n]]
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

    // [[v, t], [v, t], [v, t]]
    faces: Array<Array<Array<number>>> = [];
    // indices [v, v, v, v] per tet
    tetsIndices: Array<number> = [];
}

export class TetFile {
    // these two will go to simulation
    vertices: Float32Array;
    tetsIndices: Uint32Array;

    // this will be just copied to GL buffer
    texCoords: Float32Array;

    // this will need to be calculated after vertices update
    faceNormals: Float32Array;
    verticesNormals: Float32Array;
    // temporary buffer for normal calculation
    faceAreas: Float32Array;

    // this will be used to reconstruct GL buffer
    faces: Array<Array<Array<number>>>;
    // list of faces connected to vertex
    connectedFaces: Map<number, Array<number>>;

    // data buffer for gl buffer
    vertexBufferData: Float32Array;
    vertexBuffer: WebGLBuffer;

    triangleCount: number;

    public recalcNormals(): void {
        for (let faceIndex = 0; faceIndex < this.faces.length; faceIndex++) {
            const face = this.faces[faceIndex];

            const x0 = this.vertices[face[0][0] * 3 + 0];
            const y0 = this.vertices[face[0][0] * 3 + 1];
            const z0 = this.vertices[face[0][0] * 3 + 2];

            const x1 = this.vertices[face[1][0] * 3 + 0];
            const y1 = this.vertices[face[1][0] * 3 + 1];
            const z1 = this.vertices[face[1][0] * 3 + 2];

            const x2 = this.vertices[face[2][0] * 3 + 0];
            const y2 = this.vertices[face[2][0] * 3 + 1];
            const z2 = this.vertices[face[2][0] * 3 + 2];

            const xA = x0 - x1;
            const yA = y0 - y1;
            const zA = z0 - z1;

            const xB = x0 - x2;
            const yB = y0 - y2;
            const zB = z0 - z2;

            const xC = x1 - x2;
            const yC = y1 - y2;
            const zC = z1 - z2;

            const a = Math.hypot(xA, yA, zA);
            const b = Math.hypot(xB, yB, zB);
            const c = Math.hypot(xC, yC, zC);

            const s = (a + b + c) * 0.5;

            const area = Math.sqrt(s * (s - a) * (s - b) * (s - c));

            const xN = yA * zB - zA * yB;
            const yN = zA * xB - xA * zB;
            const zN = xA * yB - yA * xB;
            const n = Math.hypot(xN, yN, zN);

            this.faceNormals[faceIndex * 3 + 0] = xN / n;
            this.faceNormals[faceIndex * 3 + 1] = yN / n;
            this.faceNormals[faceIndex * 3 + 2] = zN / n;

            this.faceAreas[faceIndex] = area;
        }

        for (const entry of this.connectedFaces.entries()) {
            const vertexIndex = entry[0];
            const connections = entry[1];

            let vnx = 0;
            let vny = 0;
            let vnz = 0;
            let areaSum = 0;

            for (const faceIndex of connections) {
                const fnx = this.faceNormals[faceIndex * 3 + 0];
                const fny = this.faceNormals[faceIndex * 3 + 1];
                const fnz = this.faceNormals[faceIndex * 3 + 2];
                const faceArea = this.faceAreas[faceIndex];

                vnx += fnx;
                vny += fny;
                vnz += fnz;

                areaSum += faceArea;
            }

            vnx /= areaSum;
            vny /= areaSum;
            vnz /= areaSum;

            const len = Math.hypot(vnx, vny, vnz);

            vnx /= len;
            vny /= len;
            vnz /= len;

            this.verticesNormals[vertexIndex * 3 + 0] = vnx;
            this.verticesNormals[vertexIndex * 3 + 1] = vny;
            this.verticesNormals[vertexIndex * 3 + 2] = vnz;
        }
    }

    public copyDataToBuffer(gl: WebGLRenderingContext): void {
        let dataIndex = 0;
        for (let faceIndex = 0; faceIndex < this.faces.length; faceIndex++) {
            const face = this.faces[faceIndex];

            for (let faceEntryIndex = 0; faceEntryIndex < face.length; faceEntryIndex++) {
                const faceEntry = face[faceEntryIndex];

                this.vertexBufferData[dataIndex++] = this.vertices[faceEntry[0] * 3 + 0];
                this.vertexBufferData[dataIndex++] = this.vertices[faceEntry[0] * 3 + 1];
                this.vertexBufferData[dataIndex++] = this.vertices[faceEntry[0] * 3 + 2];

                this.vertexBufferData[dataIndex++] = this.verticesNormals[faceEntry[0] * 3 + 0];
                this.vertexBufferData[dataIndex++] = this.verticesNormals[faceEntry[0] * 3 + 1];
                this.vertexBufferData[dataIndex++] = this.verticesNormals[faceEntry[0] * 3 + 2];

                this.vertexBufferData[dataIndex++] = this.texCoords[faceEntry[1] * 2 + 0];
                this.vertexBufferData[dataIndex++] = this.texCoords[faceEntry[1] * 2 + 1];
            }
        }

        gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, this.vertexBufferData, gl.DYNAMIC_DRAW);
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

            const data = new Float32Array(obj.faces.length * 3 * (3 + 3 + 2));
            let dataIndex = 0;
            for (let faceIndex = 0; faceIndex < obj.faces.length; faceIndex++) {
                const face = obj.faces[faceIndex];

                for (let faceEntryIndex = 0; faceEntryIndex < face.length; faceEntryIndex++) {
                    const faceEntry = face[faceEntryIndex];

                    const vertex = obj.vertices[faceEntry[0]];
                    const texCoord = obj.texCoords[faceEntry[1]];
                    const normal = obj.normals[faceEntry[2]];

                    data[dataIndex++] = vertex[0] * size;
                    data[dataIndex++] = vertex[1] * size;
                    data[dataIndex++] = vertex[2] * size;

                    data[dataIndex++] = normal[0];
                    data[dataIndex++] = normal[1];
                    data[dataIndex++] = normal[2];

                    data[dataIndex++] = texCoord[0];
                    data[dataIndex++] = texCoord[1];
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
            result.triangleCount = tet.faces.length;

            result.vertices = new Float32Array(tet.vertices);
            result.tetsIndices = new Uint32Array(tet.tetsIndices);
            result.faces = tet.faces;
            result.texCoords = new Float32Array(tet.texCoords);
            result.faceNormals = new Float32Array(result.triangleCount * 3);
            result.verticesNormals = new Float32Array(result.triangleCount * 3 * 3);
            result.faceAreas = new Float32Array(result.triangleCount);
            result.vertexBufferData = new Float32Array(result.triangleCount * 3 * (3 + 3 + 2));

            result.connectedFaces = new Map();
            for (let faceIndex = 0; faceIndex < result.faces.length; faceIndex++) {
                for (let faceVertexIndex = 0; faceVertexIndex < 3; faceVertexIndex++) {
                    const vertexIndex = result.faces[faceIndex][faceVertexIndex][0];
                    const connections = result.connectedFaces.get(vertexIndex) || [];
                    if (!connections.includes(faceIndex)) {
                        connections.push(faceIndex);
                    }
                    result.connectedFaces.set(vertexIndex, connections);
                }
            }

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
                gl.activeTexture(gl.TEXTURE0);
                gl.bindTexture(gl.TEXTURE_2D, tex);
                gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
                gl.generateMipmap(gl.TEXTURE_2D);

                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

                resolve(tex);
            };
            image.onerror = (): void => {
                reject('Failed to load an image.');
            };
            image.src = url;
        });
    }

    async loadShader<T extends CompiledShader>(
        type: new () => T,
        baseUrl: string,
        parameters: Array<string>,
    ): Promise<T> {
        const gl = this.gl;
        return new Promise<T>(async (resolve, reject) => {
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

            const result = new type();
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
