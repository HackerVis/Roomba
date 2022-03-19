int motorPin = 9;

void setup() {
   pinMode(motorPin, OUTPUT);
   Serial.begin(9600);
}

void loop() {
         analogWrite(motorPin, 175);

   
}
