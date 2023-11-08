#include <WiFi.h>
#include <HTTPClient.h>

const char* ssid = "VIH";
const char* password = "vih_00000";

IPAddress local_IP(192, 168, 4, 4);  // Set your desired static IP address
IPAddress gateway(192, 168, 4, 1);    // Set your gateway IP address
IPAddress subnet(255, 255, 255, 0);   // Set your subnet mask

String jsonResponse;
WiFiServer server(80);

const int vibMotor = 26;

void setup() {
  Serial.begin(115200);
  pinMode(vibMotor, OUTPUT);
  delay(1000);

  WiFi.begin(ssid, password);
  Serial.println("\nConnecting");

  while(WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    delay(100);
  }

  WiFi.config(local_IP, gateway, subnet);

  Serial.println("\nConnected to the WiFi network");
  Serial.print("[+] ESP32 IP: ");
  Serial.println(WiFi.localIP());

  server.begin();
  Serial.println("Server started");
}

void loop() {
  WiFiClient client = server.available();

  if (client) {
    while (client.connected()) {
      if (client.available()) {
        String line = client.readStringUntil('\r');
        if (line.startsWith("GET /left/active")) {
          Serial.println("left active");
          jsonResponse = "{\"left\": \"active\"}";
          activateMotor();
        } else if (line.startsWith("GET /left/inactive")) {
          Serial.println("left inactive");
          jsonResponse = "{\"left\": \"inactive\"}";
          
        }
        break;
      }
    }

    client.println("HTTP/1.1 200 OK");
    client.println("Content-Type: application/json");
    client.println("Access-Control-Allow-Origin: *");
    client.println("Connection: close");
    client.print("Content-Length: ");
    client.println(jsonResponse.length());
    client.println();
    client.println(jsonResponse);

  }
}

void activateMotor(){
   digitalWrite(vibMotor, HIGH);
   delay(200);
   digitalWrite(vibMotor, LOW);
}
