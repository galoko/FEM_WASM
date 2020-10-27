import { TetFile } from './loader';
import { vec3 } from 'gl-matrix';

export default class Simulation {
    private wasm: any;
    private model: TetFile;

    private verticesPtr: number;
    private tetsIndicesPtr: number;
    private visibleIndicesPtr: number;

    public async init(): Promise<void> {
        this.wasm = await this.loadWASM('FEM');
    }

    private async loadWASM(name: string): Promise<void> {
        return new Promise<void>(function(resolve, reject: any): void {
            const initializer = require(`../wasm/${name}.js`);
            const wasm = require(`../wasm/${name}.wasm`);
            initializer({
                wasmBinary: wasm,
                print: console.log.bind(console),
            }).then(function(module: any): void {
                delete module.then;
                resolve(module);
            });
        });
    }

    public setModel(model: TetFile): void {
        /*
        this.model = model;

        const vertices = this.model.parsedFile.vertices;
        const tetsIndices = this.model.parsedFile.tets;
        const visibleIndices = this.model.visibleIndices;

        this.verticesPtr = this.wasm._malloc(vertices.length * 3 * 4);
        this.tetsIndicesPtr = this.wasm._malloc(tetsIndices.length * 4 * 4);
        this.visibleIndicesPtr = this.wasm._malloc(visibleIndices.length * 1 * 4);

        // transfer vertices directly to heap
        for (let i = 0; i < vertices.length; i++) {
            this.wasm.HEAPF32.set(vertices[i], Math.floor(this.verticesPtr / 4) + i * 3);
        }

        // transfer tets indices directly to heap
        for (let i = 0; i < tetsIndices.length; i++) {
            this.wasm.HEAPU32.set(Math.floor(this.tetsIndicesPtr / 4) + i * 4, tetsIndices[i]);
        }

        this.wasm.HEAPU32.set();
        const visibleIndicesArray = new Uint32Array(visibleIndices);

        // передаем данные в WASM

        this.wasm.HEAPF32.set(verticesArray, Math.floor(this.verticesPtr / verticesArray.BYTES_PER_ELEMENT));
        this.wasm.HEAPU32.set(tetsIndicesArray, Math.floor(this.tetsIndicesPtr / tetsIndicesArray.BYTES_PER_ELEMENT));
        this.wasm.HEAPU32.set(
            visibleIndicesArray,
            Math.floor(this.visibleIndicesPtr / visibleIndicesArray.BYTES_PER_ELEMENT),
        );

        this.wasm._setupModel(
            vertices.length,
            this.verticesPtr,
            tetsIndices.length,
            this.tetsIndicesPtr,
            visibleIndices.length,
            this.visibleIndicesPtr,
        );

        debugger;
        */
    }

    public setWallsSize(size: number): void {
        //
    }

    public process(delta: number): boolean {
        // TODO
        return true;
    }
}
