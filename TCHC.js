import {FMnistData} from './TCHC-data.js';
var canvas, ctx, saveButton, clearButton;
var pos = {x:0, y:0};
var rawImage;
var model;
 
function getModel() {
    
    // In the space below create a convolutional neural network that can classify the 
    // images of articles of clothing in the Fashion MNIST dataset. Your convolutional
    // neural network should only use the following layers: conv2d, maxPooling2d,
    // flatten, and dense. Since the Fashion MNIST has 10 classes, your output layer
    // should have 10 units and a softmax activation function. You are free to use as
    // many layers, filters, and neurons as you like.  
    // HINT: Take a look at the MNIST example.
    model = tf.sequential();
    
    // YOUR CODE HERE
    model.add(tf.layers.conv2d({inputShape: [50, 50, 1], kernelSize: 2, filters: 5, activation: 'relu'}));
    model.add(tf.layers.maxPooling2d({poolSize: [2, 2]}));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dropout({rate: 0.5}));
    model.add(tf.layers.dense({units: 512, activation: 'relu'}));
    model.add(tf.layers.dropout({rate: 0.5}));
    model.add(tf.layers.dense({units: 4, activation: 'softmax'}));

    
    // Compile the model using the categoricalCrossentropy loss,
    // the tf.train.adam() optimizer, and accuracy for your metrics.
    model.compile({optimizer: tf.train.adam(), loss: 'categoricalCrossentropy', metrics: ['accuracy']}); // YOUR CODE HERE);
    
    return model;
}

async function train(model, data) {
        
    // Set the following metrics for the callback: 'loss', 'val_loss', 'acc', 'val_acc'.
    const metrics = ['loss', 'val_loss', 'acc', 'val_acc']; // YOUR CODE HERE    

        
    // Create the container for the callback. Set the name to 'Model Training' and 
    // use a height of 1000px for the styles. 
    const container = { name: 'Model Training', styles: { height: '1000px' } };// YOUR CODE HERE   
    
    
    // Use tfvis.show.fitCallbacks() to setup the callbacks. 
    // Use the container and metrics defined above as the parameters.
    const fitCallbacks = tfvis.show.fitCallbacks(container, metrics); // YOUR CODE HERE
    
    const BATCH_SIZE = 12;
    const TRAIN_DATA_SIZE = 180;
    const TEST_DATA_SIZE = 24;
    
    // Get the training batches and resize them. Remember to put your code
    // inside a tf.tidy() clause to clean up all the intermediate tensors.
    // HINT: Take a look at the MNIST example.
    const [trainXs, trainYs] = tf.tidy(() => {
        const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
        return [
            d.xs.reshape([TRAIN_DATA_SIZE, 50, 50, 1]),
            d.labels
        ];
    });// YOUR CODE HERE

    
    // Get the testing batches and resize them. Remember to put your code
    // inside a tf.tidy() clause to clean up all the intermediate tensors.
    // HINT: Take a look at the MNIST example.
    const [testXs, testYs] = tf.tidy(() => {
        const d = data.nextTestBatch(TEST_DATA_SIZE);
        return [
            d.xs.reshape([TEST_DATA_SIZE, 50, 50, 1]),
            d.labels
        ];
    });// YOUR CODE HERE

    
    return model.fit(trainXs, trainYs, {
        batchSize: BATCH_SIZE,
        validationData: [testXs, testYs],
        epochs: 10,
        shuffle: true,
        callbacks: fitCallbacks
    });
}

function setPosition(e){
    pos.x = e.clientX-100;
    pos.y = e.clientY-100;
}
    
function draw(e) {
    if(e.buttons!=1) return;
    ctx.beginPath();
    ctx.lineWidth = 12;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'black';
    ctx.moveTo(pos.x, pos.y);
    setPosition(e);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
    rawImage.src = canvas.toDataURL('image/png');
}
    
function erase() {
    ctx.fillStyle = "white";
    ctx.fillRect(0,0,280,280);
}
    
function save() {
    var raw = tf.browser.fromPixels(rawImage,1);
    var resized = tf.image.resizeBilinear(raw, [50,50]);
    var tensor = resized.expandDims(0);
    
    var prediction = model.predict(tensor);
    var pIndex = tf.argMax(prediction, 1).dataSync();
    
    var classNames = ["人", "工", "智", "慧"];
            
            
    alert(classNames[pIndex]);
}
    
function init() {
    canvas = document.getElementById('canvas');
    rawImage = document.getElementById('canvasimg');
    ctx = canvas.getContext("2d");
    ctx.fillStyle = "white";
    ctx.fillRect(0,0,280,280);
    canvas.addEventListener("mousemove", draw);
    canvas.addEventListener("mousedown", setPosition);
    canvas.addEventListener("mouseenter", setPosition);
    saveButton = document.getElementById('sb');
    saveButton.addEventListener("click", save);
    clearButton = document.getElementById('cb');
    clearButton.addEventListener("click", erase);
}


async function run() {
    const data = new FMnistData();
    await data.load();
    console.log(data)
    const model = getModel();
    tfvis.show.modelSummary({name: 'Model Architecture'}, model);
    await train(model, data);
    // await model.save('downloads://my_model');
    init();
    alert("模型已訓練完成, 你/妳可以開始辨識手寫字體!");
}

document.addEventListener('DOMContentLoaded', run);



