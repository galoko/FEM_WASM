import { TetFile } from './loader';

type AllowedArrays = Float32Array | Uint32Array;

interface EmscriptenModule {
    HEAP8: Int8Array;
    then: Function;

    _malloc(size: number): number;
    _setupSimulation(
        verticesPtr: number,
        verticesCount: number,
        tetsIndicesPtr: number,
        tetsIndicesCount: number,
        wallsSize: number,
    ): void;
    _tick(dt: number): void;
}

export default class Simulation {
    private wasm: EmscriptenModule;
    private model: TetFile;

    private verticesPtr: number;
    private tetsIndicesPtr: number;
    private visibleIndicesPtr: number;

    private wallsSize: number;

    public async init(): Promise<void> {
        this.wasm = await this.loadWASM('FEM');
    }

    private async loadWASM(name: string): Promise<EmscriptenModule> {
        return new Promise<EmscriptenModule>(function(resolve, reject: any): void {
            const initializer = require(`../wasm/${name}.js`);
            const wasm = require(`../wasm/${name}.wasm`);
            initializer({
                wasmBinary: wasm,
                print: console.log.bind(console),
            }).then((module: EmscriptenModule): void => {
                delete module.then;
                resolve(module);
            });
        });
    }

    private allocArrayInSim<T extends AllowedArrays>(srcArray: T): T {
        const constructor = srcArray.constructor as new (
            buffer: ArrayBufferLike,
            byteOffset: number,
            length: number,
        ) => T;
        const byteSize = srcArray.length * srcArray.BYTES_PER_ELEMENT;

        const ptr = this.wasm._malloc(byteSize);

        const view = new constructor(this.wasm.HEAP8.buffer, ptr, srcArray.length);
        view.set(srcArray);

        return view;
    }

    public setModel(model: TetFile): void {
        this.model = model;

        this.model.vertices = this.allocArrayInSim(this.model.vertices);
        this.model.tetsIndices = this.allocArrayInSim(this.model.tetsIndices);
    }

    public setWallsSize(size: number): void {
        this.wallsSize = size;
    }

    public commit(): void {
        this.wasm._setupSimulation(
            this.model.vertices.byteOffset,
            this.model.vertices.length,
            this.model.tetsIndices.byteOffset,
            this.model.tetsIndices.length,
            this.wallsSize,
        );
    }

    public process(delta: number): boolean {
        this.wasm._tick(delta / 1000);
        return true;
    }
}
