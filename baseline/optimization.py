import numpy as np

from geometry import calc_side, check_overlap_single, find_corner_trees, get_global_bbox


def sa_v3(xs, ys, angs, n, iterations, T0, Tmin, move_scale, rot_scale, seed):
    np.random.seed(seed)
    bxs, bys, bangs = xs.copy(), ys.copy(), angs.copy()
    cxs, cys, cangs = xs.copy(), ys.copy(), angs.copy()
    bs = calc_side(bxs, bys, bangs, n)
    cs = bs
    T = T0
    alpha = (Tmin / T0) ** (1.0 / iterations)
    no_imp = 0
    for _ in range(iterations):
        move_type = np.random.randint(0, 8)
        sc = T / T0
        i = -1
        ox, oy, oa = 0.0, 0.0, 0.0
        if move_type < 4:
            i = np.random.randint(0, n)
            ox, oy, oa = cxs[i], cys[i], cangs[i]
            cx = np.mean(cxs[:n])
            cy = np.mean(cys[:n])
            if move_type == 0:
                cxs[i] += (np.random.random() - 0.5) * 2 * move_scale * sc
                cys[i] += (np.random.random() - 0.5) * 2 * move_scale * sc
            elif move_type == 1:
                dx, dy = cx - cxs[i], cy - cys[i]
                d = np.sqrt(dx * dx + dy * dy)
                if d > 1e-6:
                    step = np.random.random() * move_scale * sc
                    cxs[i] += dx / d * step
                    cys[i] += dy / d * step
            elif move_type == 2:
                cangs[i] += (np.random.random() - 0.5) * 2 * rot_scale * sc
                cangs[i] = cangs[i] % 360
            else:
                cxs[i] += (np.random.random() - 0.5) * move_scale * sc
                cys[i] += (np.random.random() - 0.5) * move_scale * sc
                cangs[i] += (np.random.random() - 0.5) * rot_scale * sc
                cangs[i] = cangs[i] % 360
        elif move_type == 5:
            i = np.random.randint(0, n)
            ox, oy = cxs[i], cys[i]
            gx0, gy0, gx1, gy1 = get_global_bbox(cxs, cys, cangs, n)
            bcx, bcy = (gx0 + gx1) / 2, (gy0 + gy1) / 2
            dx, dy = bcx - cxs[i], bcy - cys[i]
            d = np.sqrt(dx * dx + dy * dy)
            if d > 1e-6:
                step = np.random.random() * move_scale * sc * 0.5
                cxs[i] += dx / d * step
                cys[i] += dy / d * step
        elif move_type == 6:
            corners = find_corner_trees(cxs, cys, cangs, n)
            if len(corners) > 0:
                i = corners[np.random.randint(0, len(corners))]
                ox, oy, oa = cxs[i], cys[i], cangs[i]
                gx0, gy0, gx1, gy1 = get_global_bbox(cxs, cys, cangs, n)
                bcx, bcy = (gx0 + gx1) / 2, (gy0 + gy1) / 2
                dx, dy = bcx - cxs[i], bcy - cys[i]
                d = np.sqrt(dx * dx + dy * dy)
                if d > 1e-6:
                    step = np.random.random() * move_scale * sc * 0.3
                    cxs[i] += dx / d * step
                    cys[i] += dy / d * step
                    cangs[i] += (np.random.random() - 0.5) * rot_scale * sc * 0.5
                    cangs[i] = cangs[i] % 360
            else:
                no_imp += 1
                T *= alpha
                if T < Tmin:
                    T = Tmin
                continue
        if move_type == 4 or move_type == 7:
            no_imp += 1
            T *= alpha
            if T < Tmin:
                T = Tmin
            continue
        if i != -1 and check_overlap_single(i, cxs, cys, cangs, n):
            if move_type < 4 or move_type == 5:
                cxs[i], cys[i] = ox, oy
                if move_type != 5:
                    cangs[i] = oa
            elif move_type == 6:
                cxs[i], cys[i], cangs[i] = ox, oy, oa
            no_imp += 1
            T *= alpha
            if T < Tmin:
                T = Tmin
            continue
        ns = calc_side(cxs, cys, cangs, n)
        delta = ns - cs
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            cs = ns
            if ns < bs:
                bs = ns
                bxs[:] = cxs
                bys[:] = cys
                bangs[:] = cangs
                no_imp = 0
            else:
                no_imp += 1
        else:
            cxs[:] = bxs
            cys[:] = bys
            cangs[:] = bangs
            cs = bs
            no_imp += 1
        if no_imp > 600:
            T = min(T * 3.0, T0 * 0.7)
            no_imp = 0
        T *= alpha
        if T < Tmin:
            T = Tmin
    return bxs, bys, bangs, bs


def local_search_v3(xs, ys, angs, n, max_iter):
    bxs, bys, bangs = xs.copy(), ys.copy(), angs.copy()
    bs = calc_side(bxs, bys, bangs, n)
    pos_steps = np.array([0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002])
    rot_steps = np.array([15.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.25])
    dirs = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=np.float64)
    for _ in range(max_iter):
        improved = False
        corners = find_corner_trees(bxs, bys, bangs, n)
        for ci in range(len(corners)):
            i = corners[ci]
            for ps in pos_steps:
                for d in range(8):
                    ox, oy = bxs[i], bys[i]
                    bxs[i] += dirs[d, 0] * ps
                    bys[i] += dirs[d, 1] * ps
                    if not check_overlap_single(i, bxs, bys, bangs, n):
                        ns = calc_side(bxs, bys, bangs, n)
                        if ns < bs - 1e-10:
                            bs = ns
                            improved = True
                        else:
                            bxs[i], bys[i] = ox, oy
                    else:
                        bxs[i], bys[i] = ox, oy
            for rs in rot_steps:
                for da in [rs, -rs]:
                    oa = bangs[i]
                    bangs[i] = (bangs[i] + da) % 360
                    if not check_overlap_single(i, bxs, bys, bangs, n):
                        ns = calc_side(bxs, bys, bangs, n)
                        if ns < bs - 1e-10:
                            bs = ns
                            improved = True
                        else:
                            bangs[i] = oa
                    else:
                        bangs[i] = oa
        for i in range(n):
            in_corners = False
            for ci in range(len(corners)):
                if corners[ci] == i:
                    in_corners = True
                    break
            if in_corners:
                continue
            for ps in pos_steps:
                for d in range(8):
                    ox, oy = bxs[i], bys[i]
                    bxs[i] += dirs[d, 0] * ps
                    bys[i] += dirs[d, 1] * ps
                    if not check_overlap_single(i, bxs, bys, bangs, n):
                        ns = calc_side(bxs, bys, bangs, n)
                        if ns < bs - 1e-10:
                            bs = ns
                            improved = True
                        else:
                            bxs[i], bys[i] = ox, oy
                    else:
                        bxs[i], bys[i] = ox, oy
            for rs in rot_steps:
                for da in [rs, -rs]:
                    oa = bangs[i]
                    bangs[i] = (bangs[i] + da) % 360
                    if not check_overlap_single(i, bxs, bys, bangs, n):
                        ns = calc_side(bxs, bys, bangs, n)
                        if ns < bs - 1e-10:
                            bs = ns
                            improved = True
                        else:
                            bangs[i] = oa
                    else:
                        bangs[i] = oa
        if not improved:
            break
    return bxs, bys, bangs, bs


def perturb(xs, ys, angs, n, strength, seed):
    np.random.seed(seed)
    pxs, pys, pangs = xs.copy(), ys.copy(), angs.copy()
    num_perturb = max(1, int(n * 0.15))
    for _ in range(num_perturb):
        i = np.random.randint(0, n)
        pxs[i] += (np.random.random() - 0.5) * strength
        pys[i] += (np.random.random() - 0.5) * strength
        pangs[i] = (pangs[i] + (np.random.random() - 0.5) * 60) % 360
    for _ in range(100):
        fixed = True
        for i in range(n):
            if check_overlap_single(i, pxs, pys, pangs, n):
                fixed = False
                cx, cy = np.mean(pxs[:n]), np.mean(pys[:n])
                dx, dy = cx - pxs[i], cy - pys[i]
                d = np.sqrt(dx * dx + dy * dy)
                if d > 1e-6:
                    pxs[i] -= dx / d * 0.02
                    pys[i] -= dy / d * 0.02
                pangs[i] = (pangs[i] + np.random.random() * 20 - 10) % 360
        if fixed:
            break
    return pxs, pys, pangs


def optimize_config(n, xs, ys, angs, num_restarts, sa_iters):
    best_xs, best_ys, best_angs = xs.copy(), ys.copy(), angs.copy()
    best_side = calc_side(best_xs, best_ys, best_angs, n)
    population = [(xs.copy(), ys.copy(), angs.copy(), best_side)]
    for r in range(num_restarts):
        if r == 0:
            start_xs, start_ys, start_angs = xs.copy(), ys.copy(), angs.copy()
        elif r < len(population):
            px, py, pa, _ = population[r % len(population)]
            start_xs, start_ys, start_angs = px.copy(), py.copy(), pa.copy()
        else:
            px, py, pa, _ = population[0]
            start_xs, start_ys, start_angs = perturb(px, py, pa, n, 0.1 + 0.05 * (r % 3), 42 + r * 1000 + n)
        seed = 42 + r * 1000 + n
        oxs, oys, oangs, os = sa_v3(start_xs, start_ys, start_angs, n, sa_iters,
                                     1.0, 0.000005, 0.25, 70.0, seed)
        oxs, oys, oangs, os = local_search_v3(oxs, oys, oangs, n, 300)
        population.append((oxs.copy(), oys.copy(), oangs.copy(), os))
        population.sort(key=lambda x: x[3])
        population = population[:3]
        if os < best_side:
            best_side = os
            best_xs, best_ys, best_angs = oxs.copy(), oys.copy(), oangs.copy()
    return best_xs, best_ys, best_angs, best_side
