import { vec2, vec3, vec4, mat4, glMatrix } from 'gl-matrix';
import { Loader, ObjFile, TetFile, CompiledShader } from './loader';

export default class Render {
    private gl: WebGLRenderingContext;
    private canvas: HTMLCanvasElement;

    private cubeModel: ObjFile;
    private wallTexture: WebGLTexture;

    private donutModel: TetFile;
    private donutTexture: WebGLTexture;

    private shader: CompiledShader;

    private projection: mat4;
    private view: mat4;

    private cameraPosition: vec3;
    private cameraRotation: vec2;

    private wallsSize: number;

    constructor(canvas: HTMLCanvasElement, debug = false) {
        const attributes: object = {
            alpha: true,
            antialias: true,
            failIfMajorPerformanceCaveat: true,
            powerPreference: 'high-performance',
            preserveDrawingBuffer: false,
            stencil: false,
        };

        let gl: WebGLRenderingContext | null = canvas.getContext('webgl', attributes) as WebGLRenderingContext;
        if (!gl) gl = canvas.getContext('experimental-webgl', attributes) as WebGLRenderingContext;
        if (!gl) throw new Error('WebGL is not supported.');
        if (debug) gl = this.wrapDebug(gl);

        this.canvas = canvas;
        this.gl = gl;

        this.cameraPosition = vec3.fromValues(0, 0, 0);
        this.cameraRotation = vec2.fromValues(0, 0);

        this.projection = mat4.create();
        this.view = mat4.create();

        // setup

        gl.enable(gl.CULL_FACE);
        gl.frontFace(gl.CW);
        gl.cullFace(gl.BACK);

        gl.enable(gl.DEPTH_TEST);

        gl.clearColor(0x87 / 0xff, 0xce / 0xff, 0xeb / 0xff, 1);

        gl.activeTexture(gl.TEXTURE0);

        this.resize();
        this.setupView();
    }

    async load(): Promise<void> {
        const loader: Loader = new Loader(this.gl);

        this.cubeModel = await loader.loadObj('data/cube.obj', this.wallsSize);
        this.wallTexture = await loader.loadTexture('data/wall.png');

        this.donutModel = await loader.loadTet('data/model.tet');
        this.donutTexture = await loader.loadTexture('data/model.png');

        this.shader = await loader.loadShader('data/scene', [
            'vertexPosition',
            'vertexNormal',
            'vertexTexCoord',
            'projection',
            'view',
            'tex',
        ]);

        this.gl.useProgram(this.shader.program);
    }

    resize(): void {
        const gl = this.gl;

        gl.viewport(0.0, 0.0, this.canvas.width, this.canvas.height);

        const aspectRatio = this.canvas.width / this.canvas.height;

        const FOV = 60;

        mat4.perspective(this.projection, glMatrix.toRadian(FOV), aspectRatio, 0.1, 1000.0);
    }

    setCameraPosition(x: number, y: number, z: number): void {
        this.cameraPosition[0] = x;
        this.cameraPosition[1] = y;
        this.cameraPosition[2] = z;

        this.setupView();
    }

    // генерирует углы камеры для взгляда на точку из текущей позиции камеры
    lookAtPoint(x: number, y: number, z: number): void {
        const point = vec3.fromValues(x, y, z);

        const direction = vec3.create();
        vec3.subtract(direction, point, this.cameraPosition);
        vec3.normalize(direction, direction);

        const sinX = Math.sqrt(1 - direction[2] * direction[2]);

        this.cameraRotation[0] = Math.atan2(sinX, direction[2]);
        this.cameraRotation[1] = Math.atan2(direction[0] / sinX, direction[1] / sinX);

        this.setupView();
    }

    // переводит углы камеры в матрицу вида
    setupView(): void {
        // limit angles
        this.cameraRotation[0] = Math.min(Math.max(0.1, this.cameraRotation[0]), 3.13);
        this.cameraRotation[1] %= Math.PI * 2;

        const sinX = Math.sin(this.cameraRotation[0]);
        const cosX = Math.cos(this.cameraRotation[0]);
        const sinZ = Math.sin(this.cameraRotation[1]);
        const cosZ = Math.cos(this.cameraRotation[1]);
        const cameraDirection = vec3.fromValues(sinX * sinZ, sinX * cosZ, cosX);

        const cameraLookAt = vec3.create();
        vec3.add(cameraLookAt, this.cameraPosition, cameraDirection);

        const up = vec3.fromValues(0, 0, 1);

        mat4.lookAt(this.view, this.cameraPosition, cameraLookAt, up);
    }

    draw(): void {
        const gl = this.gl;

        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

        gl.bindTexture(gl.TEXTURE_2D, this.wallTexture);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.cubeModel.buffer);

        const SIZE_OF_VERTEX = (3 + 3 + 2) * 4;

        gl.drawArrays(gl.TRIANGLES, 0, this.cubeModel.triangleCount);

        gl.finish();
    }

    wrapDebug(context: WebGLRenderingContext): WebGLRenderingContext {
        function glEnumToString(gl: WebGLRenderingContext, value: number): string {
            // Optimization for the most common enum:
            if (value === gl.NO_ERROR) {
                return 'NO_ERROR';
            }
            for (const p in gl) {
                if ((gl as any)[p] === value) {
                    return p;
                }
            }
            return '0x' + value.toString(16);
        }

        function createGLErrorWrapper(context: WebGLRenderingContext, fname: string): () => any {
            return (...args): any => {
                const rv = (context as any)[fname].apply(context, args);
                const err = context.getError();
                if (err !== context.NO_ERROR) throw 'GL error ' + glEnumToString(context, err) + ' in ' + fname;
                return rv;
            };
        }

        const wrap: any = {};
        for (const i in context) {
            try {
                if (typeof (context as any)[i] === 'function') {
                    wrap[i] = createGLErrorWrapper(context, i);
                } else {
                    wrap[i] = (context as any)[i];
                }
            } catch (e) {}
        }
        wrap.getError = (): number => {
            return context.getError();
        };
        return wrap;
    }

    getTetModel(): TetFile {
        return this.donutModel;
    }

    updateTetModel(): void {
        //
    }

    setWallsSize(size: number): void {
        this.wallsSize = size;
    }
}
