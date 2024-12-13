def get_patients_with_head_mri_images() -> str:
    return """
    SELECT 
        pt.id, 
        pt.first, 
        pt.last,
        MAX(CASE 
            WHEN c.description = 'Stroke' THEN 1 
            ELSE 0 
        END) AS stroke_flag
    FROM procedures p
    INNER JOIN conditions c
        ON p.patient = c.patient
    INNER JOIN patients pt
        ON pt.id = p.patient
    WHERE p.description = 'Magnetic resonance imaging of head (procedure)'
      AND c.description IN ('Silent micro-hemorrhage of brain (disorder)', 'Stroke')
    GROUP BY pt.id, pt.first, pt.last
    ORDER BY pt.id
    """
