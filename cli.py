import numpy as np



def calculate_kdc(t: float, c: float, window_freq_list: list, q: float = 1.0, zones: list = None) -> float:
    e_inv_total = sum([(count * (10.0 / w) ** 3) for w, count in window_freq_list])
    intensity = c ** 1.5
    
    if not zones or len(zones) == 0:
        zones = [[0.0, 100.0, 100.0]]
        
    total_w = sum([float(z[2]) for z in zones]) or 100.0
    center_of_mass = 0.0
    effective_burst_weight = 0.0
    total_active_span = 0.0
    
    for z in zones:
        z_start, z_end, z_weight = float(z[0]), float(z[1]), float(z[2])
        z_w = (z_weight / total_w)
        span_pct = max(1.0, z_end - z_start)
        total_active_span += (span_pct / 100.0)
        local_density = max(0.05, (z_w * 100.0 / span_pct))
        midpoint = ((z_start + z_end) / 2.0) / 100.0
        center_of_mass += z_w * midpoint
        
        burst = local_density ** 0.38 if local_density >= 1.0 else local_density ** 0.20
        effective_burst_weight += z_w * burst

    pos_nerve_factor = 0.85 + 0.50 * (np.clip(center_of_mass, 0.0, 1.0) ** 1.5)
    coverage_factor = np.clip(total_active_span, 0.1, 1.0) ** 0.55
    balancing_multiplier = coverage_factor * effective_burst_weight * pos_nerve_factor
    effective_e_inv = e_inv_total * balancing_multiplier

    base = (effective_e_inv + intensity) * q
    marathon_factor = 0.75 + ((t / 75.0) ** 0.55) * 0.2
    return float((base * marathon_factor) ** 0.7)


SM_BASELINE = calculate_kdc(89, 2.0, [(101, 15), (81, 14), (61, 5), (41, 24), (40, 7)], 1.0, [[0.0, 100.0, 100.0]])

def get_elo(raw_points: float) -> float:
    diff = raw_points - SM_BASELINE
    return float(np.sign(diff) * np.log1p(np.abs(diff)) * 800.0)

def apply_cps_penalty(w: float, cps: float) -> tuple:
    """calculates motor fatigue penalty for CPS >= 17."""
    x = float(cps)
    if x < 15.0:
        return w, 0.0, 240.0 / w
    

    term1 = 14500.0 / (1.0 + np.exp(-0.22 * (x - 39.2)))

    ramp = max(0.0, x - 56.0)
    term2_3 = 15.0 * (ramp ** 2) + 576.0 * ramp
    bonus_fps = term1 + term2_3
    
    base_fps = 240.0 / max(1e-9, w)
    effective_fps = base_fps + bonus_fps
    penalized_w = 240.0 / effective_fps
    return penalized_w, bonus_fps, effective_fps

def calculate_demonstrated_skill(level_elo: float, pct: float, zones: list = None) -> float:
    if pct <= 0: return 0.0
    if pct >= 100: return float(level_elo)
    
    if not zones or len(zones) == 0:
        zones = [[0.0, 100.0, 100.0]]
        
    total_w = sum([float(z[2]) for z in zones]) or 100.0
    accumulated_weight = 0.0
    
    for z in zones:
        z_start, z_end, z_weight = float(z[0]), float(z[1]), float(z[2])
        z_w = (z_weight / total_w) * 100.0
        if pct <= z_start: continue
        elif pct >= z_end: accumulated_weight += z_w
        else: accumulated_weight += z_w * ((pct - z_start) / max(0.1, z_end - z_start))
            
    progress_frac = np.clip(accumulated_weight / 100.0, 0.0, 1.0)
    return float(np.clip(float(level_elo) * (progress_frac ** 1.12), 0.0, float(level_elo)))

def calculate_attempts(player_elo: float, level_elo: float, t: int, is_completed: bool = False) -> int:
    L, P = float(level_elo), float(player_elo)
    d = L - P

    nerve = 1.0 + ((max(0.0, float(t) - 75.0) / 60.0) ** 1.2)


    if is_completed or d <= 0:
        tier_rebeat_bonus = 0.65 * ((max(0.0, L - 2716.0) / 1514.0) ** 0.95)
        log_rebeat_base = 1.20 + 2.15 * (min(L, 2716.0) / 2716.0) + tier_rebeat_bonus
        rebeat_base = (10.0 ** log_rebeat_base) * nerve

        if d < 0:
            abs_d = abs(d)
            decay = 1.0 / (1.0 + (abs_d / 500.0) ** 1.8)
            rebeat_base *= decay
        return max(1, int(rebeat_base))


    log_att_zero = 1.55 + 3.35 * ((min(L, 2716.0) / 2716.0) ** 1.25) + 2.45 * ((max(0.0, L - 2716.0) / 1514.0) ** 0.55)
    log_mastery = 1.20 + 2.15 * (min(L, 2716.0) / 2716.0) + 0.65 * ((max(0.0, L - 2716.0) / 1514.0) ** 0.85)
    
    skill_progress_ratio = np.clip(d / max(1.0, L), 0.0, 1.0)
    current_log = log_mastery + (log_att_zero - log_mastery) * (skill_progress_ratio ** 0.55)
    calc = (10.0 ** current_log) * nerve


    if L > 6500.0:
        silent_excess = (L - 6500.0) / 400.0
        calc *= 10.0 ** (silent_excess + 1.0)

    return int(np.clip(calc, 1, 1e25))

def calculate_time_to_beat(player_elo: float, level_elo: float, attempts: int, t_seconds: int, is_completed: bool = False) -> str:
    d = max(0.0, float(level_elo - player_elo))
    survival = 0.45 if (is_completed or d == 0) else (0.25 / (1.0 + (d / 600.0) ** 1.3))
    avg_duration = max(0.6, float(t_seconds) * survival)

    total_sec = float(attempts) * avg_duration
    days = (total_sec / 3600.0) / 4.0
    years = days / 365.25

    if days < 1: return "A few hours"
    if days < 31: return f"{int(days)} days"
    if days < 365: return f"{int(days / 30.44)} months"
    if years < 70: return f"{int(years)} years"
    if years > 1e9: return f"{years / 1e9:.1f} Billion Years"
    if years > 1e6: return f"{years / 1e6:.1f} Million Years"
    return f"{int(years):,} years"
# These are very inaccurate
def get_rank_metadata(elo: float) -> str:
    if elo <= 1.0: return "Classic Easy Difficulties"
    if elo <= 1000: return "Easy Demon"
    if elo <= 1400: return "Medium Demon"
    if elo <= 1700: return "Hard Demon"
    if elo <= 2300: return "Insane Demon"
    if elo <= 2580: return "Low End Extreme Demon"
    if elo <= 3000: return "Medium Extreme Demon"
    if elo <= 3200: return "High Extreme Demon"
    if elo <= 3500: return "Very High Extreme Demon"
    if elo <= 3700: return "Top 150 Demon"
    if elo <= 4000: return "Top 50 Demon"
    if elo <= 4400: return "Top 15 Demon"
    if elo <= 4575: return "Top 5 Demon"
    if elo <= 4800: return "Top 1 Demon"
    if elo <= 5100: return "Around Grief Difficulty"
    if elo <= 5600: return "Around Aeternus Difficulty"
    if elo <= 5800: return "Lightmare Difficulty"
    if elo <= 6099: return "Human Limit For ~10 years"
    if elo <= 6500: return "Permanent Human Limit"
    if elo <= 14000: return "Impossible Level"
    return "SLL & PPLL"

def fmt_attempts(n: int) -> str:
    if n >= 1e12: return f"{n / 1e12:.2f} Trillion"
    if n >= 1e9: return f"{n / 1e9:.2f} Billion"
    if n >= 1e6: return f"{n / 1e6:.2f} Million"
    return f"{n:,}"



def parse_windows(raw_str: str) -> list:
    """Parses 'size,count | size,count' into [(size, count) tuple]"""
    pairs = []
    for chunk in raw_str.split("|"):
        chunk = chunk.strip()
        if not chunk: continue
        parts = chunk.split(",")
        pairs.append((float(parts[0].strip()), float(parts[1].strip())))
    return pairs

def parse_zones(raw_str: str) -> list:
    """Parses 'start,end,weight | ...' into [[s, e, w], tuple]"""
    zones = []
    for chunk in raw_str.split("|"):
        chunk = chunk.strip()
        if not chunk: continue
        parts = chunk.split(",")
        zones.append([float(parts[0].strip()), float(parts[1].strip()), float(parts[2].strip())])
    return zones

def main():
    while True:
        print("\n" + "=" * 55)
        print("          GD PHYSICS ENGINE - COMMAND LINE")
        print("=" * 55)
        print("[1] Calculate Level Elo & Difficulty (From Specs)")
        print("[2] Calculate Attempts & Time to Beat (Player vs Level)")
        print("[3] Apply High CPS Motor Penalty (>20 CPS)")
        print("[4] Calculate Demonstrated Skill from Run Progress %")
        print("[0] Exit")
        choice = input("\nSelect Option (0-4): ").strip()
        
        if choice == "0":
            print("\nExiting. Good luck on the grind!\n")
            break

        elif choice == "1":
            print("\n--- LEVEL ELO CALCULATOR ---")
            name = input("Level Name (optional): ").strip() or "Custom Level"
            t_str = input("Length in seconds [default 80]: ").strip()
            t = float(t_str) if t_str else 80.0
            
            c_str = input("Average CPS [default 6.0]: ").strip()
            c = float(c_str) if c_str else 6.0
            
            q_str = input("Deco multiplier q [default 1.0]: ").strip()
            q = float(q_str) if q_str else 1.0
            
            z_str = input("Multi-Zone Weights (start,end,weight | ...) [default 0,100,100]: ").strip()
            zones = parse_zones(z_str) if z_str else [[0.0, 100.0, 100.0]]
            
            print("Enter windows in format: size,count | size,count")
            print("Example: 10,50 | 5,10 | 2,3 | 1,1 (size is in 240Hz frames)")
            w_str = input("Windows: ").strip()
            
            if not w_str:
                print("Error: You must provide at least one window pair.")
                continue
            try:
                windows = parse_windows(w_str)
                raw_kdc = calculate_kdc(t, c, windows, q, zones)
                elo = get_elo(raw_kdc)
                rank = get_rank_metadata(elo)
                sm_mult = 10.0 ** (elo / 800.0)
                
                print("\n" + "-" * 50)
                print(f"RESULTS FOR: {name}")
                print(f"Raw KDC Score:        {raw_kdc:,.2f}")
                print(f"Level Elo Rating:     {elo:.1f} Elo")
                print(f"Rank Classification:  {rank}")
                if sm_mult >= 1e12:
                    mult_str = f"{sm_mult/1e12:.2f}T"
                elif sm_mult >= 1e9:
                    mult_str = f"{sm_mult/1e9:.2f}B"
                elif sm_mult >= 1e6:
                    mult_str = f"{sm_mult/1e6:.2f}M"
                else:
                    mult_str = f"{sm_mult:,.1f}"
                print(f"SM Multiplier:        {mult_str}x Stereo Madness")
                print("-" * 50)
            except Exception as e:
                print(f"Error parsing level input: {e}")

        elif choice == "2":
            print("\n--- ATTEMPTS & TIME CALCULATOR ---")
            try:
                p_elo = float(input("Enter Player Elo (e.g. 0 for beginner, 2580 for pro): ").strip())
                l_elo = float(input("Enter Level Elo (e.g. 2726 for Cata, 5098 for Grief): ").strip())
                t_str = input("Level length in seconds [default 80]: ").strip()
                t = int(float(t_str)) if t_str else 80
                
                done_str = input("Is this level already beaten / re-beat? (y/n) [default n]: ").strip().lower()
                is_done = done_str == "y"
                
                attempts = calculate_attempts(p_elo, l_elo, t, is_completed=is_done)
                time_str = calculate_time_to_beat(p_elo, l_elo, attempts, t, is_completed=is_done)
                
                mode_str = "Mastered Re-beat" if is_done else "First Clear Discovery"
                print("\n" + "-" * 50)
                print(f"Mode:               {mode_str}")
                print(f"Skill Gap:          {l_elo - p_elo:+.1f} Elo")
                print(f"Estimated Attempts: {fmt_attempts(attempts)} attempts")
                print(f"Time to Complete:   {time_str} (at 4 hours/day)")
                print("-" * 50)
            except ValueError:
                print("Error: Please enter valid numeric values for Elo and length.")

        elif choice == "3":
            print("\n--- HIGH CPS MOTOR PENALTY CONVERTER ---")
            try:
                w_str = input("Enter base window size w (in 240Hz frames, e.g. 1.0 for 240 FPS): ").strip()
                w = float(w_str) if w_str else 1.0
                cps = float(input("Enter local click speed in CPS (e.g. 26.0 or 40.0): ").strip())
                
                penalized_w, bonus_fps, effective_fps = apply_cps_penalty(w, cps)
                base_fps = 240.0 / w
                
                print("\n" + "-" * 50)
                print(f"Input CPS:             {cps:.1f} CPS")
                print(f"Base Timing:           {w:.4f}f ({base_fps:,.1f} FPS)")
                print(f"Motor Penalty Added:   +{bonus_fps:,.1f} Bonus FPS")
                print(f"Effective Timing:      {effective_fps:,.1f} FPS Equivalent")
                print(f"Penalized Window (w):  {penalized_w:.6f} frames at 240Hz")
                print(f"Density Multiplier:    {(w / penalized_w)**3:.1f}x harder per click")
                print("-" * 50)
            except ValueError:
                print("Error: Please enter valid numbers.")

        elif choice == "4":
            print("\n--- RUN DEMONSTRATED SKILL CALCULATOR ---")
            try:
                l_elo = float(input("Enter Level Elo (e.g. 2726 for Cataclysm): ").strip())
                pct = float(input("Enter percentage reached (1 - 100): ").strip())
                z_str = input("Multi-Zone Weights (start,end,weight | ...) [default 0,100,100]: ").strip()
                zones = parse_zones(z_str) if z_str else [[0.0, 100.0, 100.0]]
                
                demo_skill = calculate_demonstrated_skill(l_elo, pct, zones)
                rank = get_rank_metadata(demo_skill)
                
                print("\n" + "-" * 50)
                print(f"Run Percentage:       {pct:.1f}%")
                print(f"Demonstrated Skill:   {demo_skill:.1f} Elo")
                print(f"Rank Capability:      {rank}")
                print(f"Skill Efficiency:     {demo_skill / l_elo * 100:.1f}% of total level mastery")
                print("-" * 50)
            except ValueError:
                print("Error: Please enter valid numbers.")

if __name__ == "__main__":
    main()
