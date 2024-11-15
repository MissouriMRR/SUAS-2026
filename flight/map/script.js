const ws = new WebSocket('ws://localhost:8765');

ws.onopen = () => {
    console.log("Connected to WebSocket server");
};

ws.onmessage = (event) => {
    var myImageElement = document.getElementById('map_image');
    myImageElement.src = 'modded_map.png?rand=' + Math.random();
};

function sendMessage() {
    const message = "Update Map";
    ws.send(message);
}

ws.onclose = () => {
    console.log("Disconnected from WebSocket server");
};

setInterval(sendMessage, 2000);


// client and server need to like actually update the image