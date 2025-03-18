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
    TofCellRegion{204, 57, 251, 104, 228, 81, 2304},  // Cell 0
    TofCellRegion{157, 57, 204, 104, 181, 81, 2304},  // Cell 1
    TofCellRegion{110, 57, 157, 104, 134, 81, 2304},  // Cell 2
    TofCellRegion{63, 57, 110, 104, 87, 81, 2304},  // Cell 3
    TofCellRegion{204, 104, 251, 151, 228, 128, 2304},  // Cell 4
    TofCellRegion{157, 104, 204, 151, 181, 128, 2304},  // Cell 5
    TofCellRegion{110, 104, 157, 151, 134, 128, 2304},  // Cell 6
    TofCellRegion{63, 104, 110, 151, 87, 128, 2304},  // Cell 7
    TofCellRegion{204, 151, 251, 198, 228, 175, 2304},  // Cell 8
    TofCellRegion{157, 151, 204, 198, 181, 175, 2304},  // Cell 9
    TofCellRegion{110, 151, 157, 198, 134, 175, 2304},  // Cell 10
    TofCellRegion{63, 151, 110, 198, 87, 175, 2304},  // Cell 11
    TofCellRegion{204, 199, 251, 246, 228, 223, 2304},  // Cell 12
    TofCellRegion{157, 199, 204, 246, 181, 223, 2304},  // Cell 13
    TofCellRegion{110, 199, 157, 246, 134, 223, 2304},  // Cell 14
    TofCellRegion{63, 199, 110, 246, 87, 223, 2304},  // Cell 15
};

// Helper function to check if two rectangles overlap
inline constexpr bool rectangles_overlap(
    uint16_t x1_min, uint16_t y1_min, uint16_t x1_max, uint16_t y1_max,
    uint16_t x2_min, uint16_t y2_min, uint16_t x2_max, uint16_t y2_max) {
    return (x1_min <= x2_max && x1_max >= x2_min &&
            y1_min <= y2_max && y1_max >= y2_min);
}

// Helper function to calculate the area of overlap between two rectangles
inline constexpr uint32_t overlap_area(
    uint16_t x1_min, uint16_t y1_min, uint16_t x1_max, uint16_t y1_max,
    uint16_t x2_min, uint16_t y2_min, uint16_t x2_max, uint16_t y2_max) {
    if (!rectangles_overlap(x1_min, y1_min, x1_max, y1_max, x2_min, y2_min, x2_max, y2_max)) {
        return 0;
    }
    uint16_t overlap_width = std::min(x1_max, x2_max) - std::max(x1_min, x2_min) + 1;
    uint16_t overlap_height = std::min(y1_max, y2_max) - std::max(y1_min, y2_min) + 1;
    return overlap_width * overlap_height;
}

}  // namespace coralmicro