// tft_setup.h - display setup for THIS sketch only. TFT_eSPI reads tft_setup.h from the sketch folder, so the
// library's own User_Setup.h (set up for another board) stays untouched.
// The old box used TFT_eSPI with its global User_Setup.h: if its driver or pins differ, copy those lines here.
// The screen layout adapts to any size and rotation (see ui.ino). TODO_VERIFY driver + pins on the box.
#define USER_SETUP_INFO "aqi_node WROOM-32"

#define ST7789_DRIVER                // 2.8" 240 x 320 panel. For an ILI9341 panel: ILI9341_DRIVER
#define TFT_WIDTH 240
#define TFT_HEIGHT 320
#define TFT_RGB_ORDER TFT_BGR        // if red and blue look swapped: TFT_RGB

// VSPI, the usual ESP32 wiring
#define TFT_MOSI 23
#define TFT_SCLK 18
#define TFT_CS 15
#define TFT_DC 2
#define TFT_RST 4

#define LOAD_GLCD                    // font 1 (8 px)
#define LOAD_FONT2                   // font 2 (16 px)
#define LOAD_FONT4                   // font 4 (26 px)
#define LOAD_FONT6                   // font 6 (48 px, digits only)
#define SPI_FREQUENCY 40000000
