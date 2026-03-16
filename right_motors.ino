//right motors

#define motor_1 9
#define motor_2 10


// Motor 1 direction control pins
#define motor_1_dir_1 3
#define motor_1_dir_2 4

// Motor 2 direction control pins
#define motor_2_dir_1 5
#define motor_2_dir_2 6


void setup()
{
  // Begin serial communication
  Serial.begin(9600);
  
  attachInterrupt(digitalPinToInterrupt(2) , serviceRoutine , LOW);

  // Set all the pins from 2 to 13 as output pins
  for(int i = 3; i <= 13; i++){
    pinMode(i, OUTPUT);
  }
}

// Main program loop
void loop()
{
  Forward(); // move forward
  delay(5000); // wait for 5 seconds
  Stop(); // stop moving
  delay(500); // wait for 0.5 seconds
  
  Backward(); // move backward
  delay(5000); // wait for 5 seconds
  Stop(); // stop moving
  delay(500); // wait for 0.5 seconds
  
  TurnLeft(); // turn left with no speed
  delay(5000); // wait for 5 seconds
  Stop(); // stop moving
  delay(500); // wait for 0.5 seconds
  
  TurnRight(); // turn right with no speed
  delay(5000); // wait for 5 seconds
  Stop(); // stop moving
  delay(500); // wait for 0.5 seconds
  
}

// Motor control functions
// Move the car forward
void Forward()
{
  // Enable all the motors
  Enable1();  

  // Set the direction pins for motor 1 and 2 to forward
  digitalWrite(motor_1_dir_1, HIGH);
  digitalWrite(motor_1_dir_2, LOW);
  digitalWrite(motor_2_dir_1, HIGH);
  digitalWrite(motor_2_dir_2, LOW);
  
}

// Move the car backward
void Backward()
{
  // Enable all the motors
  Enable1();

  // Set the direction pins for motor 1 and 2 to backward
  digitalWrite(motor_1_dir_1, LOW);
  digitalWrite(motor_1_dir_2, HIGH);
  digitalWrite(motor_2_dir_1, LOW);
  digitalWrite(motor_2_dir_2, HIGH);
  
  
}

// Turn the car left with given speed
void TurnLeft()
{
  // Enable all the motors
  Enable2();

  // Set the direction pins for motor 1 and 2 to forward
  digitalWrite(motor_1_dir_1, HIGH);
  digitalWrite(motor_1_dir_2, LOW);
  digitalWrite(motor_2_dir_1, HIGH);
  digitalWrite(motor_2_dir_2, LOW);

}

// Turn the car right with given speed
void TurnRight()
{
  // Enable all the motors
  Enable2();

  
  // Set the direction pins for motor 1 and 2 to forward
  digitalWrite(motor_1_dir_1, LOW);
  digitalWrite(motor_1_dir_2, HIGH);
  digitalWrite(motor_2_dir_1, LOW);
  digitalWrite(motor_2_dir_2, HIGH);
  
}


// Sets the speed of all motors to maximum
void Enable1(){
  analogWrite(motor_1, 255);
  analogWrite(motor_2, 255);
}


void Enable2(){
  analogWrite(motor_1, 123);
  analogWrite(motor_2, 123);
}


// Sets the direction pins of all four motors to stop
void Stop()
{
  digitalWrite(motor_1_dir_1,LOW);
  digitalWrite(motor_1_dir_2,LOW);
  digitalWrite(motor_2_dir_1,LOW);
  digitalWrite(motor_2_dir_2,LOW);

}

void serviceRoutine()
{
  Stop();
}