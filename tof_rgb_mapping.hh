// Auto-generated TOF cell to RGB pixel mapping
#pragma once
#include <array>
#include <cstdint>

namespace coralmicro {

// Number of TOF cells
constexpr size_t kTofCellCount = 16;  // 4x4 grid

struct TofCellRegion {
    uint16_t x_min;
    uint16_t y_min;
    uint16_t x_max;
    uint16_t y_max;
    uint16_t center_x;
    uint16_t center_y;
    uint32_t area;      // Pre-calculated area of the cell region
};

// Mapping of TOF cell regions (bounds, center, and area)
constexpr std::array<TofCellRegion, kTofCellCount> kTofCellRegions = {
    { 204, 57, 251, 104, 228, 81, 2304 },  // Cell 0
    { 157, 57, 204, 104, 181, 81, 2304 },  // Cell 1
    { 110, 57, 157, 104, 134, 81, 2304 },  // Cell 2
    { 63, 57, 110, 104, 87, 81, 2304 },  // Cell 3
    { 204, 104, 251, 151, 228, 128, 2304 },  // Cell 4
    { 157, 104, 204, 151, 181, 128, 2304 },  // Cell 5
    { 110, 104, 157, 151, 134, 128, 2304 },  // Cell 6
    { 63, 104, 110, 151, 87, 128, 2304 },  // Cell 7
    { 204, 151, 251, 198, 228, 175, 2304 },  // Cell 8
    { 157, 151, 204, 198, 181, 175, 2304 },  // Cell 9
    { 110, 151, 157, 198, 134, 175, 2304 },  // Cell 10
    { 63, 151, 110, 198, 87, 175, 2304 },  // Cell 11
    { 204, 199, 251, 246, 228, 223, 2304 },  // Cell 12
    { 157, 199, 204, 246, 181, 223, 2304 },  // Cell 13
    { 110, 199, 157, 246, 134, 223, 2304 },  // Cell 14
    { 63, 199, 110, 246, 87, 223, 2304 },  // Cell 15
}};


}  // namespace coralmicro