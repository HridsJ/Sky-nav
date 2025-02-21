var locations = [];

fetch("https://myuser.pythonanywhere.com/process", {
    method: "POST",
    headers: {
        "Content-Type": "application/json"
    },
    body: JSON.stringify({
        elements: window.inputElements  // Use inputs from initial input file
    })
})
.then(response => response.json())  // Convert response to JSON
.then(data => {
    locations = data.locations;  // Assuming response contains a "locations" array
    for (let i = 0; i < locations.length; i++) {
        let [lat, lon] = locations[i];
        console.log(`Latitude: ${lat}, Longitude: ${lon}`);
    }
})
.catch(error => console.error("Error:", error));