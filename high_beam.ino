const int LDR_PIN = A0;  // LDR module connected to A0
const int THRESHOLD = 795; // Threshold for detecting high beam

void setup() {
    Serial.begin(9600); // Start serial communication
    pinMode(LDR_PIN, INPUT);
}

void loop() {
    int light_value = analogRead(LDR_PIN);  // Read LDR value
    //Serial.println(light_value);  // Send raw value to Serial Monitor

    if (1023-light_value > THRESHOLD) {
        Serial.println("1"); // High beam detected
    } else {
        Serial.println("0"); // No high beam
    }
    
    delay(500); // Small delay to avoid spamming
}