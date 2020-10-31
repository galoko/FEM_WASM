import Render from './render';
import Simulation from './simulation';

async function main(): Promise<void> {
    const wallsSize = 4.3;

    const canvas: HTMLCanvasElement = document.getElementById('scene') as HTMLCanvasElement;

    const render: Render = new Render(canvas, false);
    render.setWallsSize(wallsSize);
    await render.load();

    render.setCameraPosition(2, 2, 2);
    render.lookAtPoint(0, 0, 0);

    const simulation = new Simulation();
    await simulation.init();

    simulation.setModel(render.getTetModel());
    simulation.setWallsSize(wallsSize);
    simulation.commit();

    // main loop
    let lastTime: number | undefined = undefined;
    const draw = (time: number): void => {
        const delta: number = time - (lastTime || time);
        const stepped = simulation.process(delta);
        if (stepped) {
            render.updateTetModel();
        }
        render.resize();
        render.draw();
        lastTime = time;

        requestAnimationFrame(draw);
    };
    requestAnimationFrame(draw);
}

main();
