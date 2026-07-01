try:
    from .base_annotator import BasePoseVisualizer
except ImportError:
    from base_annotator import BasePoseVisualizer

class COCOVisualizer(BasePoseVisualizer):
    """Configuration for COCO 17-keypoint format."""
    
    HEX_COLORS = {
        'head': '#1B00FF', 'torso': '#E203FF', 
        'lower_body': '#36FF2B', 'connector': '#FF8000'
    }
    
    HEAD_KPTS = {0, 1, 2, 3, 4}
    LOWER_BODY_KPTS = {13, 14, 15, 16}
    HEAD_TO_TORSO = {(3, 5), (4, 6)}

    def _define_skeleton(self):
        return [
            (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 7), (7, 9),
            (6, 8), (8, 10), (5, 6), (0, 1), (0, 2), (1, 3), (2, 4), (5, 11),
            (6, 12), (3, 5), (4, 6)
        ]

    def _assign_keypoint_colors(self):
        colors = []
        bgr = {k: self._hex_to_bgr(v) for k, v in self.HEX_COLORS.items()}
        
        for i in range(17):
            if i in self.HEAD_KPTS: c = bgr['head']
            elif i in self.LOWER_BODY_KPTS: c = bgr['lower_body']
            else: c = bgr['torso']
            colors.append(c)
        return colors

    def _assign_limb_colors(self):
        colors = []
        bgr = {k: self._hex_to_bgr(v) for k, v in self.HEX_COLORS.items()}
        connector_limbs = [tuple(sorted(x)) for x in self.HEAD_TO_TORSO]
        
        for p1, p2 in self.skeleton:
            pair = tuple(sorted((p1, p2)))
            if pair in connector_limbs: c = bgr['connector']
            elif p1 in self.HEAD_KPTS and p2 in self.HEAD_KPTS: c = bgr['head']
            elif p1 in self.LOWER_BODY_KPTS and p2 in self.LOWER_BODY_KPTS: c = bgr['lower_body']
            else: c = bgr['torso']
            colors.append(c)
        return colors
