class CanalDescriptor:
    def __init__(self):
        # Water level descriptions
        self.water_remarks = {
            1: ["Canal is dry, requires immediate attention.", 
                "No water flow detected.",
                "Critical: Water supply interrupted."],
            2: ["Water level below operational requirements.", 
                "Low water flow observed.",
                "Water level needs adjustment."],
            3: ["Water level within normal range.", 
                "Optimal water flow maintained.",
                "Standard operating conditions."],
            4: ["High water level detected.", 
                "Near overflow conditions.",
                "Monitor closely for overflow risk."],
            5: ["Critical: Canal overflowing.", 
                "Immediate action required: Overflow condition.",
                "Emergency: Water exceeds canal capacity."]
        }

        # Silt level descriptions (starting from 2)
        self.silt_remarks = {
            2: ["Minor silt accumulation present.", 
                "Light sediment buildup observed.",
                "Minimal siltation detected."],
            3: ["Normal silt levels present.", 
                "Typical sediment conditions.",
                "Acceptable siltation level."],
            4: ["Significant silt buildup.", 
                "Maintenance needed: High siltation.",
                "Silt affecting canal efficiency."],
            5: ["Critical siltation level.", 
                "Urgent dredging required.",
                "Severe sediment accumulation."]
        }

        # Debris level descriptions (starting from 2)
        self.debris_remarks = {
            2: ["Minor debris present.", 
                "Light scattered debris observed.",
                "Limited obstruction by debris."],
            3: ["Normal debris presence.", 
                "Manageable debris level.",
                "Standard debris conditions."],
            4: ["Heavy debris accumulation.", 
                "Significant debris blockage.",
                "Debris affecting flow."],
            5: ["Canal blocked by debris.", 
                "Critical debris obstruction.",
                "Urgent debris removal needed."]
        }

    def get_remark(self, water_level, silt_level, debris_level):
        """Generate a combined remark based on condition levels."""
        import random

        # Get random remarks for each condition
        water_remark = random.choice(self.water_remarks.get(int(water_level), 
                                   ["Water level status unknown."]))
        silt_remark = random.choice(self.silt_remarks.get(int(silt_level), 
                                  ["Normal silt conditions."]))
        debris_remark = random.choice(self.debris_remarks.get(int(debris_level), 
                                    ["Normal debris conditions."]))

        # Determine overall condition
        conditions = []
        if water_level in [1, 5]:  # Critical water levels
            conditions.append("CRITICAL")
        elif water_level in [2, 4]:  # Concerning water levels
            conditions.append("WARNING")
            
        if silt_level >= 4:
            conditions.append("MAINTENANCE REQUIRED")
            
        if debris_level >= 4:
            conditions.append("CLEANUP NEEDED")

        # Create status line
        status = " | ".join(conditions) if conditions else "NORMAL OPERATION"

        # Combine remarks
        combined_remark = (
            f"STATUS: {status}\n"
            f"Water: {water_remark}\n"
            f"Silt: {silt_remark}\n"
            f"Debris: {debris_remark}"
        )

        return combined_remark

    def get_short_remark(self, water_level, silt_level, debris_level):
        """Generate a shorter, one-line remark for quick assessment."""
        conditions = []
        
        # Add critical conditions only
        if water_level == 1:
            conditions.append("DRY")
        elif water_level == 5:
            conditions.append("OVERFLOW")
        elif water_level == 4:
            conditions.append("HIGH WATER")
            
        if silt_level >= 4:
            conditions.append("SILTED")
            
        if debris_level >= 4:
            conditions.append("BLOCKED")
            
        if not conditions:
            return "Normal operating conditions."
        
        return "Alert: " + " + ".join(conditions)