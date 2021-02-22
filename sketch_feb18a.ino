
//modified version of:
///* How to use the DHT-22 sensor with Arduino uno
//   Temperature and humidity sensor
//   More info: http://www.ardumotive.com/how-to-use-dht-22-sensor-en.html
//   Dev: Michalis Vasilakis // Date: 1/7/2015 // www.ardumotive.com */
//   Modified: 3/12/2019
//   Modified: 2/18/2021



//Libraries
#include <DHT.h>

//Constants
#define DHTPIN1 4    
#define DHTPIN2 7


#define DHTTYPE DHT22   
DHT dht1(DHTPIN1, DHTTYPE); 
DHT dht2(DHTPIN2, DHTTYPE); 



#define relay1 11 
#define relay2 12


//Variables
float setTemp = 24.00;
float setHum = 35.00;

int chk;

float hum1;  
float temp1; 
float hum2;  
float temp2; 


float H1b;  
float T1b; 
float H2b;  
float T2b; 


float hum1Dif = (H1b - hum1);
float temp1Dif = (T1b - temp1);
float hum2Dif = (H2b - hum2);
float temp2Dif = (T2b - temp2);


long randNumber77;

void setup()
{
  delay(1000);
  
  Serial.begin(9600);
  
  delay(1000);
  
  dht1.begin();
  dht2.begin();

  
  pinMode (relay1, OUTPUT);
  pinMode (relay2, OUTPUT);
  
  digitalWrite(relay1, HIGH);
  digitalWrite(relay2, HIGH);
  
  delay(1000);

}




////////////





void loop()

  {
    delay(6000);
    randNumber77 = random(11,14);
    digitalWrite(randNumber77, HIGH);
    
    delay(1000);

    hum1 = dht1.readHumidity();
    temp1 = dht1.readTemperature();
    hum2 = dht2.readHumidity();
    temp2 = dht2.readTemperature();

    delay(1000);
    
    H1b = dht1.readHumidity();
    T1b = dht1.readTemperature();
    H2b = dht2.readHumidity();
    T2b = dht2.readTemperature();

    
    delay(1000);

    hum1 = dht1.readHumidity();
    temp1 = dht1.readTemperature();
    hum2 = dht2.readHumidity();
    temp2 = dht2.readTemperature();

    delay(1000);
    
    H1b = dht1.readHumidity();
    T1b = dht1.readTemperature();
    H2b = dht2.readHumidity();
    T2b = dht2.readTemperature();

    
    delay(1000);


           //    //    //    101 & 102    //    //    //
            
      if (temp1 >= setTemp || temp2 >= setTemp){ //(hum1 > setHum || hum2 > setHum || hum3 > setHum || hum4 > setHum || hum5 > setHum) {
    
        hum1 = dht1.readHumidity();
        temp1 = dht1.readTemperature();
        hum2 = dht2.readHumidity();
        temp2 = dht2.readTemperature();

        
        delay(1000);
   
        hum1 = dht1.readHumidity();
        temp1 = dht1.readTemperature();
        hum2 = dht2.readHumidity();
        temp2 = dht2.readTemperature();

        
        delay(1000);

        hum1 = dht1.readHumidity();
        temp1 = dht1.readTemperature();
        hum2 = dht2.readHumidity();
        temp2 = dht2.readTemperature();

        
        delay(1000);
    
        Serial.print(hum1);
        Serial.print(",");
        Serial.print(temp1);
        Serial.print(",");
        Serial.print(hum2);
        Serial.print(",");    
        Serial.print(temp2);
        Serial.print(",");

        
        delay(1000);

        Serial.print(randNumber77);
        
        Serial.print(",1,");
        
        digitalWrite (randNumber77, LOW);
                    
        delay(4000);                    //run Load
    
        Serial.print("101,");           //101 is for section of this arduino sketch, it helps with troubleshooting later to find out if a certain area of the programs is working or not

        delay(1000);
        
        H1b = dht1.readHumidity();
        T1b = dht1.readTemperature();
        H2b = dht2.readHumidity();
        T2b = dht2.readTemperature();

    
        delay(1000);
        
        H1b = dht1.readHumidity();
        T1b = dht1.readTemperature();
        H2b = dht2.readHumidity();
        T2b = dht2.readTemperature();

    
        delay(1000); 
               
        H1b = dht1.readHumidity();
        T1b = dht1.readTemperature();
        H2b = dht2.readHumidity();
        T2b = dht2.readTemperature();

    
        delay(1000);
        
        Serial.print(H1b);
        Serial.print(",");
        Serial.print(T1b);
        Serial.print(",");    
        Serial.print(H2b);
        Serial.print(",");
        Serial.print(T1b);
        

                
        float hum1Dif = (H1b - hum1);
        float temp1Dif = (T1b - temp1);
        float hum2Dif = (H2b - hum2);
        float temp2Dif = (T2b - temp2);


        Serial.print(",0,");
        
        Serial.print(hum1Dif);
        Serial.print(",");
        Serial.print(temp1Dif);
        Serial.print(",");    
        Serial.print(hum2Dif);
        Serial.print(",");
        Serial.println(temp2Dif);


        delay(1000);
        
        }
                else{
                        digitalWrite(randNumber77, HIGH);
                        
                        delay(1000);
    
                        hum1 = dht1.readHumidity();
                        temp1 = dht1.readTemperature();
                        hum2 = dht2.readHumidity();
                        temp2 = dht2.readTemperature();

                        
                        delay(1000);
                    
                        hum1 = dht1.readHumidity();
                        temp1 = dht1.readTemperature();
                        hum2 = dht2.readHumidity();
                        temp2 = dht2.readTemperature();


                        delay(1000);
                    
                        Serial.print(hum1);
                        Serial.print(",");
                        Serial.print(temp1);
                        Serial.print(",");
                        Serial.print(hum2);
                        Serial.print(",");    
                        Serial.print(temp2);
                        Serial.print(",");

                        
                        delay(1000);
                        
                        Serial.print(randNumber77);
                        
                        Serial.print(",0,");              // "0" means off, "1" means on, used for determining which data points correlate with the temperature and humidity at a particular location
                            
                        delay(3000);                    //run Load
                    
                        Serial.print("102,");
                    
                        H1b = dht1.readHumidity();
                        T1b = dht1.readTemperature();
                        H2b = dht2.readHumidity();
                        T2b = dht2.readTemperature();
                    
                        delay(1000);

                        
                        H1b = dht1.readHumidity();
                        T1b = dht1.readTemperature();
                        H2b = dht2.readHumidity();
                        T2b = dht2.readTemperature();

                    
                        delay(1000);
                        
                        H1b = dht1.readHumidity();
                        T1b = dht1.readTemperature();
                        H2b = dht2.readHumidity();
                        T2b = dht2.readTemperature();

                    
                        delay(1000);
                                
                        Serial.print(H1b);
                        Serial.print(",");
                        Serial.print(T1b);
                        Serial.print(",");    
                        Serial.print(H2b);
                        Serial.print(",");
                        Serial.print(T2b);

                        
                        float hum1Dif = (H1b - hum1);
                        float temp1Dif = (T1b - temp1);
                        float hum2Dif = (H2b - hum2);
                        float temp2Dif = (T2b - temp2);

                
                        Serial.print(",0,");
                        
                        Serial.print(hum1Dif);
                        Serial.print(",");
                        Serial.print(temp1Dif);
                        Serial.print(",");    
                        Serial.print(hum2Dif);
                        Serial.print(",");
                        Serial.println(temp2Dif);

                                    
                        delay(1000);
                        
                        }


                                  








hum1 = dht1.readHumidity();
temp1 = dht1.readTemperature();
hum2 = dht2.readHumidity();
temp2 = dht2.readTemperature();


delay(1000);

            
            
            //    //    //    103 & 104    //    //    //
            
      if (temp1 >= setTemp || temp2 >= setTemp) { 
    
        hum1 = dht1.readHumidity();
        temp1 = dht1.readTemperature();
        hum2 = dht2.readHumidity();
        temp2 = dht2.readTemperature();

        
        delay(1000);
   
        hum1 = dht1.readHumidity();
        temp1 = dht1.readTemperature();
        hum2 = dht2.readHumidity();
        temp2 = dht2.readTemperature();

        
        delay(1000);
        
        Serial.print(hum1);
        Serial.print(",");
        Serial.print(temp1);
        Serial.print(",");
        Serial.print(hum2);
        Serial.print(",");    
        Serial.print(temp2);
        Serial.print(",");
        
        delay(1000);

        Serial.print(randNumber77);
        
        Serial.print(",1,");
                    
        delay(4100);                    //run Load
        
        Serial.print("103,");
    
        H1b = dht1.readHumidity();
        T1b = dht1.readTemperature();
        H2b = dht2.readHumidity();
        T2b = dht2.readTemperature();

    
        delay(1000);
        
        H1b = dht1.readHumidity();
        T1b = dht1.readTemperature();
        H2b = dht2.readHumidity();
        T2b = dht2.readTemperature();

    
        delay(1000);

        H1b = dht1.readHumidity();
        T1b = dht1.readTemperature();
        H2b = dht2.readHumidity();
        T2b = dht2.readTemperature();


                
        Serial.print(H1b);
        Serial.print(",");
        Serial.print(T1b);
        Serial.print(",");    
        Serial.print(H2b);
        Serial.print(",");
        Serial.print(T2b);

        
        float hum1Dif = (H1b - hum1);
        float temp1Dif = (T1b - temp1);
        float hum2Dif = (H2b - hum2);
        float temp2Dif = (T2b - temp2);


        Serial.print(",0,");
        
        Serial.print(hum1Dif);
        Serial.print(",");
        Serial.print(temp1Dif);
        Serial.print(",");    
        Serial.print(hum2Dif);
        Serial.print(",");
        Serial.println(temp2Dif);


        delay(1000);

        
        }

                else{
                        digitalWrite(randNumber77, HIGH);
                        
                        delay(1000);
                    
                        hum1 = dht1.readHumidity();
                        temp1 = dht1.readTemperature();
                        hum2 = dht2.readHumidity();
                        temp2 = dht2.readTemperature();

                        
                        delay(1000);
                   
                        hum1 = dht1.readHumidity();
                        temp1 = dht1.readTemperature();
                        hum2 = dht2.readHumidity();
                        temp2 = dht2.readTemperature();
                        
                        delay(1000);
                
         
                        Serial.print(hum1);
                        Serial.print(",");
                        Serial.print(temp1);
                        Serial.print(",");
                        Serial.print(hum2);
                        Serial.print(",");    
                        Serial.print(temp2);
                        Serial.print(",");

                        
                        delay(1000);
                        
                        Serial.print(randNumber77);
                        
                        Serial.print(",0,");
                            
                        delay(4100);                    //run Load
                    
                        Serial.print("104,");
                    
                        H1b = dht1.readHumidity();
                        T1b = dht1.readTemperature();
                        H2b = dht2.readHumidity();
                        T2b = dht2.readTemperature();

                    
                        delay(1000);
                        
                        H1b = dht1.readHumidity();
                        T1b = dht1.readTemperature();
                        H2b = dht2.readHumidity();
                        T2b = dht2.readTemperature();

                    
                        delay(1000);
                                
                        Serial.print(H1b);
                        Serial.print(",");
                        Serial.print(T1b);
                        Serial.print(",");    
                        Serial.print(H2b);
                        Serial.print(",");
                        Serial.print(T2b);

                        
                        float hum1Dif = (H1b - hum1);
                        float temp1Dif = (T1b - temp1);
                        float hum2Dif = (H2b - hum2);
                        float temp2Dif = (T2b - temp2);

                
                        Serial.print(",0,");
                        
                        Serial.print(hum1Dif);
                        Serial.print(",");
                        Serial.print(temp1Dif);
                        Serial.print(",");    
                        Serial.print(hum2Dif);
                        Serial.print(",");
                        Serial.println(temp2Dif);

                                    
                        delay(1000);
            
                        }                                                            


        
      hum1 = dht1.readHumidity();
      temp1 = dht1.readTemperature();
      hum2 = dht2.readHumidity();
      temp2 = dht2.readTemperature();

      
      delay(1000);
 



            //    //    //    105 & 106    //    //    //
            
      if (temp1 >= setTemp || temp2 >= setTemp ){    
        
        hum1 = dht1.readHumidity();
        temp1 = dht1.readTemperature();
        hum2 = dht2.readHumidity();
        temp2 = dht2.readTemperature();
    
        
        delay(1000);
   
        hum1 = dht1.readHumidity();
        temp1 = dht1.readTemperature();
        hum2 = dht2.readHumidity();
        temp2 = dht2.readTemperature();
    
        
        delay(1000);
        
        Serial.print(hum1);
        Serial.print(",");
        Serial.print(temp1);
        Serial.print(",");
        Serial.print(hum2);
        Serial.print(",");    
        Serial.print(temp2);
        Serial.print(",");
    
        
        delay(1000);

        Serial.print(randNumber77);
        
        Serial.print(",1,");
        digitalWrite(randNumber77, LOW);            
        delay(4100);                    //run Load
    
        Serial.print("105,");
    
        H1b = dht1.readHumidity();
        T1b = dht1.readTemperature();
        H2b = dht2.readHumidity();
        T2b = dht2.readTemperature();
    
    
        delay(1000);
        
        H1b = dht1.readHumidity();
        T1b = dht1.readTemperature();
        H2b = dht2.readHumidity();
        T2b = dht2.readTemperature();
    
    
        delay(1000);
                
        Serial.print(H1b);
        Serial.print(",");
        Serial.print(T1b);
        Serial.print(",");    
        Serial.print(H2b);
        Serial.print(",");
        Serial.print(T2b);
    
        
        float hum1Dif = (H1b - hum1);
        float temp1Dif = (T1b - temp1);
        float hum2Dif = (H2b - hum2);
        float temp2Dif = (T2b - temp2);
    

        Serial.print(",0,");
        
        Serial.print(hum1Dif);
        Serial.print(",");
        Serial.print(temp1Dif);
        Serial.print(",");    
        Serial.print(hum2Dif);
        Serial.print(",");
        Serial.println(temp2Dif);
    

        delay(1000);}

                else{
                        digitalWrite(randNumber77, HIGH);
                        
                        delay(1000);
    
                        hum1 = dht1.readHumidity();
                        temp1 = dht1.readTemperature();
                        hum2 = dht2.readHumidity();
                        temp2 = dht2.readTemperature();
    
                        
                        delay(1000);
                    
                        hum1 = dht1.readHumidity();
                        temp1 = dht1.readTemperature();
                        hum2 = dht2.readHumidity();
                        temp2 = dht2.readTemperature();
    

                        delay(1000);
                    
                        Serial.print(hum1);
                        Serial.print(",");
                        Serial.print(temp1);
                        Serial.print(",");
                        Serial.print(hum2);
                        Serial.print(",");
                        Serial.print(temp2);
                        Serial.print(",");    
    
                        
                        delay(1000);
                        
                        Serial.print(randNumber77);
                        
                        Serial.print(",0,");
                            
                        delay(4100);                    //run Load
                    
                        Serial.print("106,");
                    
                        H1b = dht1.readHumidity();
                        T1b = dht1.readTemperature();
                        H2b = dht2.readHumidity();
                        T2b = dht2.readTemperature();
    
                    
                        delay(1000);
                        
                        H1b = dht1.readHumidity();
                        T1b = dht1.readTemperature();
                        H2b = dht2.readHumidity();
                        T2b = dht2.readTemperature();
    
                    
                        delay(1000);
                                
                        Serial.print(H1b);
                        Serial.print(",");
                        Serial.print(T1b);
                        Serial.print(",");    
                        Serial.print(H2b);
                        Serial.print(",");
                        Serial.print(T2b);
    
                        
                        float hum1Dif = (H1b - hum1);
                        float temp1Dif = (T1b - temp1);
                        float hum2Dif = (H2b - hum2);
                        float temp2Dif = (T2b - temp2);
    
                
                        Serial.print(",0,");
                        
                        Serial.print(hum1Dif);
                        Serial.print(",");
                        Serial.print(temp1Dif);
                        Serial.print(",");    
                        Serial.print(hum2Dif);
                        Serial.print(",");
                        Serial.println(temp2Dif);
    
                                    
                        delay(1000);
            
                        }   
                          
                          
                          
     
      hum1 = dht1.readHumidity();
      temp1 = dht1.readTemperature();
      hum2 = dht2.readHumidity();
      temp2 = dht2.readTemperature();
    
      
      delay(1000);
                    

            
                        return;}   
                          
                                 



        
