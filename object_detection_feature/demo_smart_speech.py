#!/usr/bin/env python3
"""
Demo script showing smart speech capabilities
"""

from smart_detector import SmartObjectDetector

def demo_smart_speech():
    """Demonstrate smart speech capabilities"""
    
    print("🤖 Smart Object Detection with Environment Understanding")
    print("=" * 60)
    print()
    print("This system understands object relationships and speaks naturally!")
    print()
    print("Examples of what it can say:")
    print("• 'I can see a person holding a bowl' (instead of 'person bowl')")
    print("• 'I can see a person sitting on a chair'")
    print("• 'I can see a person using a laptop'")
    print("• 'I can see a laptop on a table'")
    print("• 'I can see a person wearing a hat'")
    print("• 'I can see a person carrying a bag'")
    print()
    print("Press 'p' to hear smart speech about what it sees!")
    print("Press 'q' to quit")
    print()
    
    # Create detector with smart speech
    detector = SmartObjectDetector(
        model_size='s',      # Small model for good balance
        confidence=0.4,      # Moderate confidence
        nms_threshold=0.4,   # Standard NMS
        max_detections=15    # Reasonable limit
    )
    
    # Run the detector
    detector.run()

if __name__ == "__main__":
    demo_smart_speech() 