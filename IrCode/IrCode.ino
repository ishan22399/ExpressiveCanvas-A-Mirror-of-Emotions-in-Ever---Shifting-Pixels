#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>

const char* ssid = "#";
const char* password = "#";

ESP8266WebServer server(80);

const int irSensorPin = D5;  // Connect IR sensor to digital pin D5
bool irSensorActive = false;

void handleStartIR() {
    irSensorActive = true;
    server.send(200, "text/plain", "IR sensor started");
}

void handleStopIR() {
    irSensorActive = false;
    server.send(200, "text/plain", "IR sensor stopped");
}

void handleStatus() {
    int motion = digitalRead(irSensorPin);
    String response = "{\"motion\":" + String(motion) + "}";
    server.send(200, "application/json", response);
}

void setup() {
    Serial.begin(115200);
    pinMode(irSensorPin, INPUT);
    WiFi.begin(ssid, password);

    while (WiFi.status() != WL_CONNECTED) {
        delay(1000);
        Serial.println("Connecting to WiFi...");
    }
    Serial.println("Connected to WiFi");

    server.on("/start-ir", handleStartIR);
    server.on("/stop-ir", handleStopIR);
    server.on("/status", handleStatus);

    server.begin();
    Serial.println("HTTP server started");
}

void loop() {
    server.handleClient();

    if (irSensorActive) {
        int motion = digitalRead(irSensorPin);
        if (motion == HIGH) {
            Serial.println("Motion detected");
            // Here you can add the code to handle motion detection
        }
    }
}
