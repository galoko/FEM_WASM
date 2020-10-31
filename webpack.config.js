const path = require('path');

module.exports = {
    entry: './src/ts/main.ts',
    module: {
        rules: [
            {
                test: /\.tsx?$/,
                use: 'ts-loader',
                exclude: /node_modules/,
            },
            {
                test: /\.wasm$/,
                type: 'javascript/auto',
                loader: 'arraybuffer-loader',
                options: {
                    name: '[name]-[hash].[ext]',
                },
            },
        ],
    },
    node: {
        fs: 'empty',
    },
    resolve: {
        extensions: ['.tsx', '.ts', '.js', '.c', '.cpp'],
    },
    output: {
        filename: 'main.js',
        path: path.resolve(__dirname, 'build'),
        publicPath: '/',
    },
    devServer: {
        contentBase: 'build',
        host: '0.0.0.0',
        hot: true,
    },
    optimization: {
        namedModules: true,
    },
};
