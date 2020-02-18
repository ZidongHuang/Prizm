import codeanticode.syphon.*;
import processing.video.*;
import java.awt.*;
import java.util.*;
import java.util.Random;
import java.lang.Math;
import gab.opencv.*;

// parameters
int canvas_width = 1378/2;
int canvas_height = 861/2;
int colomn_left = 617/2;
int colomn_top = 101/2;
int colomn_bottom = 821/2;
int colomn_right = 760/2;

int MAXTIMEOUT = 20;              // facial recogition maximutime out
int maxpeople = 100;              // maximum people support
int maxparticles = 100;           // particle number
int random_particle_connection = 20;      // random particles connection distance
int rad = 5;                      // circle radius
int vertex_rad = 2;               // colomn vertex radius
int rancircle_rad = 1;            // random circle radius
float y_speed_normal = 1;         // circle speed
float motion_ratio = 0.05;         // circle motion ratio (user controlled)
float face_rad_ratio = 0.2;        // face radius ratio
//int color_stack_x = colomn_left;          // color stack x position (always fixed)
//int color_stack_width = colomn_right - colomn_left;       // each color stack: width
//int color_stack_height = rad;     // each color stack: height

// colomn vertex
int vertex_num = 44;
int v_x[] = {352, 325, 368, 378, 309, 311, 376, 311, 322, 339,
              380, 365, 311, 309, 311, 376, 310, 309, 321, 336,
              354, 371, 379, 378, 310, 309, 380, 310, 320, 337, 
              358, 372, 376, 313, 313, 307, 378, 358, 377, 331, 
              377, 322, 355, 310}; 
int v_y[] = {50, 55, 55, 60, 62, 77, 106, 128, 130, 131,
              130, 134, 134, 139, 148, 163, 198, 208, 211, 206,
              208, 213, 210, 267, 266, 280, 286, 289, 290, 291,
              293, 288, 309, 327, 329, 359, 360, 363, 365, 368, 
              401, 406, 410, 219};

// color triangles
int triangle_max = 41;
float triangle_x1[] = 
        {331, 377, 377, 307, 358, 372, 337, 380, 337, 
        372, 320, 380, 310, 310, 321, 309, 336, 354, 336, 
        309, 354, 336, 379, 376, 380, 365, 322, 365, 339, 
        380, 322, 322, 311, 312, 309, 352, 352, 315, 352, 352, 368};
float triangle_y1[] = 
        {368, 365, 365, 359, 363, 288, 291, 286, 291, 
        288, 290, 286, 289, 219, 211, 208, 206, 208, 206, 
        208, 208, 206, 210, 163, 130, 134, 130, 134, 131, 
        130, 130, 130, 128, 77, 62, 50, 50, 55, 50, 50, 55};
float triangle_x2[] = 
        {322, 355, 331, 322, 331, 331, 307, 358, 331, 
        358, 337, 372, 313, 310, 358, 337, 358, 372, 321, 
        321, 380, 372, 371, 371, 354, 321, 309, 336, 321, 
        354, 321, 311, 309, 311, 322, 339, 365, 309, 315, 380, 378};
float triangle_y2[] = 
        {406, 410, 368, 406, 368, 368, 359, 363, 368, 
        293, 291, 288, 327, 289, 293, 291, 293, 288, 211, 
        211, 286, 288, 213, 213, 208, 211, 208, 206, 211, 
        208, 211, 134, 139, 134, 130, 131, 134, 62, 55, 130, 60};
float triangle_x3[] = 
        {355, 377, 355, 331, 377, 358, 331, 377, 358, 
        331, 307, 358, 320, 320, 337, 320, 372, 380, 358, 
        337, 371, 354, 378, 379, 371, 336, 321, 354, 365, 
        365, 339, 310, 311, 322, 339, 365, 380, 339, 339, 368, 376};
float triangle_y3[] = 
        {410, 401, 410, 368, 365, 363, 368, 365, 293, 
        368, 359, 363, 290, 290, 291, 290, 288, 286, 293, 
        291, 213, 208, 267, 210, 213, 206, 211, 208, 134, 
        134, 131, 198, 148, 130, 131, 134, 130, 131, 131, 55, 106};
int color_stack_idx = 0;
int color_stack_current_height = 410;             // initial bottom
// each valid color stack
ArrayList<Integer> color_stack_valid = new ArrayList<Integer>();
// each color stack: RGB color
ArrayList<Integer> color_stack_color_r = new ArrayList<Integer>();
ArrayList<Integer> color_stack_color_g = new ArrayList<Integer>();
ArrayList<Integer> color_stack_color_b = new ArrayList<Integer>();

// variables
Capture cam;
OpenCV opencv;
PGraphics canvas;
System particles;
SyphonServer server;

// motion recogition
int numPixels;
float []previousFrame;
float pixAverage;
boolean motion_flag;
int prev_x = canvas_width/2;
int prev_y = canvas_height/2;
int diff = 0;                      // for circle speed control

// facial recogition
boolean face_flag;
int face_timeout = 0;
double face_rad = 0;

// circle
int color_r_possible[] = {10,254,1,252,251,252,252,254,10,251,1,10,254,254,1,252,252,10,1,251,252,252,254,251,10,31,252,31,10,252,252,1,10,254,10,251,10,252,252,254,251,10};
int color_g_possible[] = {253,254,200,7,11,7,7,254,253,11,200,253,254,254,200,7,7,253,200,11,7,7,254,11,253,173,7,173,253,7,108,200,253,254,253,11,253,7,108,24,11,253};
int color_b_possible[] = {145,0,219,248,100,248,248,0,145,100,219,145,0,0,219,248,248,145,219,100,248,248,0,100,145,219,248,219,145,248,12,219,145,0,145,100,145,248,12,0,100,145};
int color_number = 42;
int color_idx = 0;
float [] xpos = new float[maxpeople];                     // all <maxpeople> circles: x position
float [] ypos = new float[maxpeople];                     // all <maxpeople> circles: x position
int [] active_shape = new int[maxpeople];                 // all <maxpeople> circles: active flag (1 for active)
int [] circle_color_r = new int [maxpeople];              // all <maxpeople> circle: color r
int [] circle_color_g = new int [maxpeople];              // all <maxpeople> circle: color g
int [] circle_color_b = new int [maxpeople];              // all <maxpeople> circle: color b
int last_active_shape = -1;                                // last active circle (if not time out, user still has control)

void setup() {
  size(689, 430, P3D);
  canvas = createGraphics(canvas_width, canvas_height, P3D);
  server = new SyphonServer(this, "Processing Syphon");   // create syhpon server to send frames out.

  String[] cameras = Capture.list();
  if (cameras.length == 0) {
    println("There are no cameras available for capture.");
    exit();
  } 
  else {
    println("Available cameras:");
    for (int i = 0; i < cameras.length; i++) {
      println(cameras[i]);
    }
    
    cam = new Capture(this, canvas_width, canvas_height, cameras[0]);
    opencv = new OpenCV(this, canvas_width, canvas_height);
    opencv.loadCascade(OpenCV.CASCADE_FRONTALFACE);
    cam.start(); 
    numPixels = cam.width * cam.height;
    previousFrame = new float[numPixels];
    loadPixels();
    stroke(255, 255, 255);
    strokeWeight(1);
    frameRate(10);
    ellipseMode(RADIUS);
    for (int i = 0; i < maxpeople; i++) {
      xpos[i] = (colomn_left + colomn_right) / 2;
      ypos[i] = colomn_top + rad; 
      active_shape[i] = 0; 
    }
    // add random particles
    particles = new System();
    for (int i = 0; i < maxparticles; i++) {
      particles.addCircle();
    }
    for (int i = 0; i < vertex_num; i++) {
      particles.addVertex(v_x[i], v_y[i]);
    }
    particles.update();
    // add colomn vertex
  }
}

void draw() {  
  cam.read();
  opencv.loadImage(cam);
  cam.loadPixels();
  noFill();
  
  int r = color_r_possible[color_idx];
  int g = color_g_possible[color_idx];
  int b = color_b_possible[color_idx];
  
  Rectangle[] faces = opencv.detect();
  if (faces.length == 0) face_timeout += 1;
  int max_face_area = 0;
  face_rad = 0;
  for (int i = 0; i < faces.length; i++) {
    stroke(255, 255, 255);
    rect(faces[i].x, faces[i].y, faces[i].width, faces[i].height);
    if (faces[i].width * faces[i].height > max_face_area) max_face_area = faces[i].width * faces[i].height;
  }
  if (max_face_area > 0) face_rad = Math.sqrt(max_face_area) * face_rad_ratio;

  // if there is no face, everything (active circles) keeps going with normal speed (1)
  if (faces.length == 0) {
    for (int i = 0; i < maxpeople; i++) {
      if (active_shape[i] == 1) ypos[i] = ypos[i] + y_speed_normal;
    }
  }
  
  // if there is a face here
  else {
    // first, calculate the motion difference and set motion_flag
    int x = 0;
    int y = 0;
    int sum = 0;
    for (int i = 0; i < numPixels; i++) {
      float currColor = red(cam.pixels[i]);
      float prevColor = previousFrame[i];
      float d = abs(prevColor-currColor);
      if (d > 50) {
        int xt = i % cam.width;
        int yt = i / cam.width;
        x += xt;
        y += yt;
        sum ++;
      }
      previousFrame[i] = currColor;
    }
    if (sum > 100) { // motion detected, set motion_flag
      x /= sum;
      y /= sum; 
      diff = abs(x - prev_x + y - prev_y);
      motion_flag = true;
      
      drawTarget(x, y);
      prev_x = x;
      prev_y = y;
    }
    else {
      diff = 0;
      motion_flag = false;
    }
    
    // if motion detected, and doesn't time out, move the last_active circle
    if (motion_flag && face_timeout < MAXTIMEOUT) {
        if (last_active_shape < 0) {
          last_active_shape += 1;
          color_idx = (color_idx + 1) % color_number;     // next circle has different color
          active_shape[last_active_shape] = 1;
          circle_color_r[last_active_shape] = r;
          circle_color_g[last_active_shape] = g;
          circle_color_b[last_active_shape] = b;
        } 
        ypos[last_active_shape] = ypos[last_active_shape] + diff * motion_ratio;
        //println(last_active_shape, ypos[last_active_shape]);
    }
    // if motion detected, but already timed out, reset time out and enable a new circle
    else if (motion_flag) {
        face_timeout = 0;
        last_active_shape += 1;
        color_idx = (color_idx + 1) % color_number;     // next circle has different color
        active_shape[last_active_shape] = 1;
        circle_color_r[last_active_shape] = r;
        circle_color_g[last_active_shape] = g;
        circle_color_b[last_active_shape] = b;
        ypos[last_active_shape] = ypos[last_active_shape] + diff * motion_ratio;
        //println("time_out", last_active_shape, ypos[last_active_shape]);
    }
    // all other active circle should keep going with normal speed (1)
    for (int i = 0; i < maxpeople; i++) {
        if (active_shape[i] == 1 && i != last_active_shape) ypos[i] = ypos[i] + y_speed_normal;
    }
  }
   
  // draw...
  canvas.beginDraw();
  canvas.stroke(255);
  canvas.background(0);
  canvas.fill(r, g, b);  // set color for the circle
  
  canvas.strokeWeight(0.4);
  particles.random_interact();
  particles.update();
  
  canvas.strokeWeight(1.5);
  for (int i = 0; i < maxpeople; i++) {
    if (active_shape[i] == 1 && ypos[i] <= color_stack_current_height - rad) {
      // circle in the air, draw it
      canvas.stroke(circle_color_r[i], circle_color_g[i], circle_color_b[i]);
      canvas.fill(circle_color_r[i], circle_color_g[i], circle_color_b[i]);
      canvas.ellipse(xpos[i], ypos[i], rad, rad);
      stroke(circle_color_r[i], circle_color_g[i], circle_color_b[i]);
      ellipse(xpos[i], ypos[i], rad, rad);
      particles.interact(xpos[i], ypos[i], face_rad, circle_color_r[i], circle_color_g[i], circle_color_b[i]);
    }
    else if (active_shape[i] == 1 && ypos[i] > color_stack_current_height - rad) {
      active_shape[i] = 0; ypos[i] = rad;  // touch the bottom color stack, disable it
      color_stack_idx += 1;
      color_stack_color_r.add(circle_color_r[i]);
      color_stack_color_g.add(circle_color_g[i]);
      color_stack_color_b.add(circle_color_b[i]);
      if (color_stack_idx > triangle_max) color_stack_idx = triangle_max;
      color_stack_current_height = (int) triangle_y1[color_stack_idx-1];
    }
  }
  // draw all available color stacks
  for (int i = 0; i < color_stack_idx; i++)  {
    canvas.stroke(color_stack_color_r.get(i), color_stack_color_g.get(i), color_stack_color_b.get(i));
    canvas.fill(color_stack_color_r.get(i), color_stack_color_g.get(i), color_stack_color_b.get(i));
    canvas.triangle(triangle_x1[i], triangle_y1[i], triangle_x2[i], triangle_y2[i], triangle_x3[i], triangle_y3[i]);
    //canvas.rect(color_stack_x, color_stack_y.get(i), color_stack_width, color_stack_height); 
    stroke(color_stack_color_r.get(i), color_stack_color_g.get(i), color_stack_color_b.get(i));
    triangle(triangle_x1[i], triangle_y1[i], triangle_x2[i], triangle_y2[i], triangle_x3[i], triangle_y3[i]);
    //rect(color_stack_x, color_stack_y.get(i), color_stack_width, color_stack_height);
  }
  canvas.endDraw();
  image(canvas, 0, 0);
  image(cam, 0, 0);
  server.sendImage(canvas);
}

// draw moving target
void drawTarget(int x, int y)
{
  stroke(255, 255, 255);
  line(x,y-15 , x,y-4);
  line(x,y+15 , x,y+4);
  line(x-15,y , x-4,y);
  line(x+15,y , x+4,y);
}

class Body {
  PVector pos, speed, accln;
  float mass;
  float drag = 0.999;
  
  Body(PVector pos_, PVector speed_, PVector accln_, float mass_){
    pos = pos_;
    speed = speed_;
    accln =  accln_;
    mass = mass_;
  }
  
  void update(){
    speed.add(accln);
    pos.add(speed);
    speed.mult(drag);
    
   // bounce off the walls
    if (pos.x<colomn_left){speed.x = -speed.x;}
    if (pos.x>colomn_right){speed.x = -speed.x;}
    if (pos.y<colomn_top){speed.y = -speed.y;}
    if (pos.y>colomn_bottom){speed.y = -speed.y;}
    accln.set(0,0,0);
  }

  
//speed of the movement of particles default is five
  void applyForce(PVector force){
    force.mult(5/mass);   
    accln.add(force);
  }
  
  void changeSpeed(PVector newSpeed){
    speed = newSpeed;
  }
  
  void render(){
  }
}
class Circle extends Body {
  
  float r;  
  Circle(PVector pos_, PVector speed_, PVector accln_, float mass_, float r_){
    super(pos_, speed_, accln_, mass_);
    r = r_;
  }
  
  void render(){
    ellipseMode(RADIUS);
    fill(128, 128, 128);                           // processing particle color: gray
    ellipse(pos.x, pos.y, r, r);
  }
  
  void update(){
    super.update();
  }
  
  void applyForce(PVector force){
    super.applyForce(force);
  }
  
  void changeSpeed(PVector newSpeed){
    //super.changeSpeed(newSpeed);
  }
}
class Vertex extends Body {
  
  float r;  
  Vertex(PVector pos_, float r_){
    super(pos_, new PVector(0, 0), new PVector(0, 0), 0.0);
    r = r_;
  }
  
  void render(){
    ellipseMode(RADIUS);
    fill(255, 255, 255);                           // processing vertex color
    ellipse(pos.x, pos.y, r, r);
    //canvas.fill(255, 255, 255);                           // processing vertex color
    //canvas.ellipse(pos.x, pos.y, r, r);
  }
  
  void update(){
    super.update();
  }
  
  void applyForce(PVector force){
    //super.applyForce(force);
  }
  
  void changeSpeed(PVector newSpeed){
    //super.changeSpeed(newSpeed);
  }
}
class System {
  ArrayList circles;
  ArrayList vertexs;
  
  System() {
    circles = new ArrayList();
    vertexs = new ArrayList();
  }
  
  void addCircle(){
    Circle p = new Circle(new PVector(random(colomn_left,colomn_right),random(colomn_top,colomn_bottom)),
        new PVector(random(-1,1),random(-1,1)),
        new PVector(0,0), 10, rancircle_rad);
    circles.add(p);
  }  
  
  void addVertex (int x, int y) {
    Vertex p = new Vertex(new PVector(x, y), vertex_rad);
    vertexs.add(p);
  }
  void update() {
    for (int i = circles.size() - 1; i >= 0; i-- ) {
      Circle p = (Circle) circles.get(i);
      p.update();
      p.render();
    }
    for (int i = vertexs.size() - 1; i > 0; i --) {
      Vertex p = (Vertex) vertexs.get(i);
      p.update();
      p.render();
    }
  }
  
  void interact(float x, float y, double r, int color_red, int color_green, int color_blue){
    PVector target = new PVector(x, y);
    for (int i = circles.size() - 1; i >= 0; i-- ) {
      Circle p = (Circle) circles.get(i);
        PVector temp = PVector.sub(p.pos, target);
        if (temp.mag() < r){
          stroke(color_red, color_green, color_blue);               // processing circle line color
          canvas.stroke(color_red, color_green, color_blue);        // project circle line color
          line(p.pos.x,p.pos.y, target.x,target.y);
          canvas.line(p.pos.x,p.pos.y, target.x,target.y);
        }
    }
    for (int i = vertexs.size() - 1; i >= 0; i--) {
      Vertex p = (Vertex) vertexs.get(i);
        PVector temp = PVector.sub(p.pos, target);
        if (temp.mag() < r){
          stroke(color_red, color_green, color_blue);               // processing circle line color
          canvas.stroke(color_red, color_green, color_blue);        // project circle line color
          line(p.pos.x,p.pos.y, target.x,target.y);
          canvas.line(p.pos.x,p.pos.y, target.x,target.y);
        }
    }
  }
  void random_interact(){
    // interact with random particles
    for (int i = circles.size() - 1; i >= 0; i-- ) {
      Circle p = (Circle) circles.get(i);
      for (int j = i; j >= 0; j-- ) {
        Circle q = (Circle) circles.get(j);
        PVector temp = PVector.sub(p.pos,q.pos);
        
         //length of the line
         //background 20, moving point 50<n<80
               
        if (temp.mag()<20 & temp.mag()>1){    
          temp.normalize();
          temp.mult(0.1);
          p.applyForce(temp);
          temp.mult(-1);
          q.applyForce(temp);
          stroke(128, 128, 128);               // processing particle line color
          canvas.stroke(255, 255, 255);        // project particle line color
          line(p.pos.x,p.pos.y,q.pos.x,q.pos.y);
          canvas.line(p.pos.x,p.pos.y,q.pos.x,q.pos.y);
        }
      }
    }
  }
}
