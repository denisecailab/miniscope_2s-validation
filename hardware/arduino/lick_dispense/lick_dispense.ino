#include <Wire.h>
#include "Adafruit_MPR121.h"

#ifndef _BV
#define _BV(bit) (1 << (bit)) 
#endif

int sensors[] = {0, 1};

int valves[] = {2, 3};

unsigned long tstart;
unsigned long treset;
unsigned long pumpOpen = 400;
bool isRewarding = false;
int rewardingPort = 0;
byte lastDispense = 255;
uint16_t curr_touched;

int nSensors = sizeof(sensors) / sizeof(int);

Adafruit_MPR121 cap = Adafruit_MPR121();
int rwCount = 0;
int incomingByte = 0;

void setup() {
  Serial.begin(115200);

  while (!Serial);

  Serial.println("Serial initialized.");

  // Default address is 0x5A at 5V. 
  // If tied to 3.3V it's 0x5B. 
  while(1){
    if (cap.begin(0x5A)) {
      break;
      }
  }
  Serial.println("Sensor found!");

  for (byte i = 0; i < nSensors; i++){
    pinMode(valves[i], OUTPUT);
    digitalWrite(valves[i], LOW);
  }

  cap.setThresholds(3, 2);
}

void loop() {
  // get capacitive sensor input. 
  curr_touched = cap.touched();
  
  for (byte i = 0; i < nSensors; i++){
    if ((curr_touched & _BV(i)) && (i != lastDispense) && (!isRewarding)){
        digitalWrite(valves[i], HIGH);
        isRewarding = true;
        lastDispense = i;
        rewardingPort = valves[i];
        tstart = millis();
        rwCount += 1;
        Serial.print("p:");
        Serial.print(i);
        Serial.print("r:");
        Serial.println(rwCount);
    }
  }
  if (isRewarding && (millis() - tstart > pumpOpen)){
    digitalWrite(rewardingPort, LOW);
    isRewarding = false;
  }
  if (Serial.available() > 0) {
    incomingByte = Serial.read();
    if (incomingByte == 114) {
      rwCount = 0;
      lastDispense = 255;
      Serial.println("Reward Reset");
    }
  }
}
