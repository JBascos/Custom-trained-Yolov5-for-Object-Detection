
#include <WiFi.h>

// Replace with your network credentials
const char* ssid     = "VIH";
const char* password = "vih_00000";

// Set web server port number to 80
WiFiServer server(80);

// Variable to store the HTTP request
String header;

const int vibMotor = 26;

void setup() {
  Serial.begin(115200);
    // Initialize the output variables as output

  // Connect to Wi-Fi network with SSID and password
  Serial.print("Setting AP (Access Point)…");
  // Remove the password parameter, if you want the AP (Access Point) to be open
  WiFi.softAP(ssid, password);

  IPAddress IP = WiFi.softAPIP();
  Serial.print("AP IP address: ");
  Serial.println(IP);

  pinMode(vibMotor,OUTPUT);
  
  server.begin();
}

void loop(){
  WiFiClient client = server.available();   // Listen for incoming clients


  if (client) {                             // If a new client connects,
    Serial.println("New Client.");          // print a message out in the serial port
    String currentLine = "";                // make a String to hold incoming data from the client
    String jsonResponse = "";
    while (client.connected()) {            // loop while the client's connected
      if (client.available()) {             // if there's bytes to read from the client,
        char c = client.read();             // read a byte, then
        Serial.write(c);                    // print it out the serial monitor
        header += c;
        if (c == '\n') {                    // if the byte is a newline character
          // if the current line is blank, you got two newline characters in a row.
          // that's the end of the client HTTP request, so send a response:
          if (currentLine.length() == 0) {
            // HTTP headers always start with a response code (e.g. HTTP/1.1 200 OK)
            // and a content-type so the client knows what's coming, then a blank line:
            
            // turns the GPIOs on and off
            if (header.indexOf("GET /right/active") >= 0) {
              Serial.println("right active");
              jsonResponse = "{\"right\": \"active\"}";
              activateMotor();
            } else if (header.indexOf("GET /right/inactive") >= 0) {
              Serial.println("right inactive");
              jsonResponse = "{\"right\": \"inactive\"}";
            } 
            
            client.println("HTTP/1.1 200 OK");
            client.println("Content-Type: application/json");
            client.println("Access-Control-Allow-Origin: *");
            client.println("Connection: close");
            client.print("Content-Length: ");
            client.println(jsonResponse.length());
            client.println();
            client.println(jsonResponse);

            Serial.println(jsonResponse);
            
            // Break out of the while loop
            break;
          } else { // if you got a newline, then clear currentLine
            currentLine = "";
          }
        } else if (c != '\r') {  // if you got anything else but a carriage return character,
          currentLine += c;      // add it to the end of the currentLine
        }
      }
    }
    // Clear the header variable
    header = "";
    // Close the connection
    client.stop();
    Serial.println("Client disconnected.");
    Serial.println("");
  }
}


void activateMotor(){
   digitalWrite(vibMotor, HIGH);
   delay(200);
   digitalWrite(vibMotor, LOW);
}
