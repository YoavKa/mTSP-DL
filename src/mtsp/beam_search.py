import torch

from ..utils import USE_GPU, zeros, ones, is_sorted


class PermutationBeamSearch:
    def __init__(self, log_probs, groups, use_gpu=USE_GPU):
        # log_probs:    Float(groups x cities)
        # groups:       int
        self.log_probs = log_probs
        self.groups = groups
        self.cities = log_probs.size(1)

        self.minus_infinity = abs(self.log_probs.min()) * -10

        self.use_gpu = use_gpu

        self.cities_range = torch.arange(0, self.cities).long()
        self.cities_eye_byte = torch.eye(self.cities).byte()
        self.cities_eye_float = torch.eye(self.cities).float()
        if self.use_gpu:
            self.cities_range = self.cities_range.cuda()
            self.cities_eye_byte = self.cities_eye_byte.cuda()
            self.cities_eye_float = self.cities_eye_float.cuda()

    def run(self, k):
        used = results = log_probs = None

        for i in range(self.groups):
            if i == 0:
                used = self.cities_eye_byte
                results = self.cities_range.unsqueeze(1)
                log_probs = self.log_probs[0]
            else:
                used, results, log_probs = self.loop(used, results, log_probs, i)

            # leave only samples with positive probability
            ok_idx = torch.arange(0, log_probs.size(0)).long()
            if self.use_gpu:
                ok_idx = ok_idx.cuda()
            ok_idx = ok_idx[log_probs > self.minus_infinity]

            used = used[ok_idx]
            results = results[ok_idx]
            log_probs = log_probs[ok_idx]

            if log_probs.size(0) > k >= 0:
                # choose the most promising results
                top_idx = log_probs.topk(k=min(k, log_probs.size(0)), dim=0, sorted=False)[1]
                used = used[top_idx]
                results = results[top_idx]
                log_probs = log_probs[top_idx]

            # normalize the probabilities
            log_probs = log_probs - log_probs.max()

        return results

    def loop(self, used, results, log_probs, cur_group):
        # used:         Byte(beam x cities)
        # results:      Long(beam x groups_so_far)
        # log_probs:    Float(beam)
        # cur_group:    int
        beam = log_probs.size(0)

        # Long(beam x cities x groups_so_far)
        new_results = results.unsqueeze(1).expand(beam, self.cities, results.size(1))
        # Long(beam x cities x (groups_so_far + 1))
        new_results = torch.cat((new_results, self.cities_range.unsqueeze(0).expand(beam, self.cities).unsqueeze(2)),
                                dim=2)

        new_log_probs = log_probs.unsqueeze(1).expand(beam, self.cities)  # Float(beam x cities)
        new_log_probs_delta = self.log_probs[cur_group].unsqueeze(0).expand_as(new_log_probs)  # Float(beam x cities)
        new_log_probs = new_log_probs + new_log_probs_delta

        new_used = used.unsqueeze(1).expand(beam, self.cities, self.cities)  # Byte(beam x cities x cities)
        new_used = new_used.index_select(dim=1, index=self.cities_range)

        # zero probs if new cur city is already used
        # Float(beam x cities x cities)
        new_probs_used = self.cities_eye_float.unsqueeze(0).repeat(beam, 1, 1).mul_(new_used.float())
        new_log_probs[new_probs_used.sum(dim=2) == 1] = self.minus_infinity

        new_used[self.cities_eye_byte.unsqueeze(0).expand_as(new_used)] = 1

        return new_used.contiguous().view(-1, self.cities), \
            new_results.contiguous().view(-1, cur_group + 1), \
            new_log_probs.contiguous().view(-1)


class MTSPBeamSearch:
    _channels_order_ok = None

    @classmethod
    def check_channels_order(cls):
        if cls._channels_order_ok is not None:
            return cls._channels_order_ok

        # noinspection PyArgumentList
        range_ = torch.arange(3) + 1
        first = (range_ * 1).unsqueeze(0).unsqueeze(1).expand(3, 3, 3)
        second = (range_ * 10).unsqueeze(0).unsqueeze(2).expand(3, 3, 3)
        third = (range_ * 100).unsqueeze(1).unsqueeze(2).expand(3, 3, 3)
        ok = is_sorted((first + second + third).view(-1))

        cls._channels_order_ok = ok
        return ok

    def __init__(self, probs, dists):
        if not self.check_channels_order():
            raise RuntimeError('This class assumes a different channel order then the one defined!')

        # probs:        Float(groups x cities x cities)
        # dists:        Float(cities x cities)
        self.log_probs = probs.log()
        self.groups = probs.size(0)
        self.cities = probs.size(1)
        self.dists = dists

        self.minus_infinity = abs(self.log_probs.min()) * -10

        # noinspection PyArgumentList
        self.cities_range = torch.arange(self.cities).long()
        if USE_GPU:
            self.cities_range = self.cities_range.cuda()

        self.permutation_search = PermutationBeamSearch(self.log_probs[:, 0, 1:], self.groups, USE_GPU)

    def run(self, k):
        starts = cur_city = cur_group = used = results = dists = log_probs = None

        for i in range(self.cities):
            if starts is None:
                starts, cur_city, cur_group, used, results, dists, log_probs = self.init_beam(k)
            else:
                starts, cur_city, cur_group, used, results, dists, log_probs = \
                    self.loop(starts, cur_city, cur_group, used, results, dists, log_probs, k, i == self.cities - 1)

        # change the results format from Long(beam x c x 2) to Byte(beam x n x n)
        final_results = zeros(results.size(0), self.cities, self.cities, type='byte')  # Byte(beam x n x n)
        final_results = final_results.view(results.size(0), -1)  # Byte(beam x (n*n))
        from_cities = results[:, :, 0]  # Long(beam x c)
        to_cities = results[:, :, 1]  # Long(beam x c)
        final_results.scatter_(dim=1, index=from_cities * self.cities + to_cities,
                               src=ones(*from_cities.size(), type='byte'))
        final_results = final_results.view(-1, self.cities, self.cities)
        results = final_results

        return results, dists

    def init_beam(self, k):
        # starts:       Long(beam x (groups + 1))
        # cur_city:     Long(beam)
        # cur_group:    Long(beam)
        # used:         Byte(beam x cities)
        # results:      Long(beam x c x 2)
        # dists:        Float(beam x groups)
        # log_probs:    Float(beam)

        if self.cities > 200:
            all_perms = self.permutation_search.run(k=k) + 1  # Long(beam x groups)
        else:
            all_perms = self.permutation_search.run(k=10000) + 1  # Long(beam x groups)
        beam = all_perms.size(0)

        # calculate starts
        starts = torch.cat((all_perms, zeros(beam, 1, type='long')), dim=1)  # Long(beam x (groups + 1))

        # calculate cur_city
        cur_city = all_perms[:, 0]  # Long(beam)

        # calculate cur_group
        cur_group = zeros(beam, type='long')  # Long(beam)

        # calculate used
        used = zeros(beam, self.cities, type='byte')  # Byte(beam x cities)
        used.scatter_(dim=1, index=all_perms, src=ones(beam, self.groups, type='byte'))

        # calculate results
        results = zeros(beam, self.groups, 2, type='long')  # Long(beam x groups x 2)
        results[:, :, 1] = all_perms

        # calculate dists
        # Float(beam x groups)
        dists = self.dists[0, :].unsqueeze(0).expand(beam, self.cities).gather(dim=1, index=all_perms)

        # calculate log_probs
        # Float(beam x groups x cities)
        beam_probs = self.log_probs[:, 0, :].unsqueeze(0).expand(beam, self.groups, self.cities)
        log_probs = beam_probs.gather(dim=2, index=all_perms.unsqueeze(2)).squeeze(2).prod(dim=1)  # Float(beam)

        # leave only samples with positive probability
        ok_idx = (log_probs > self.minus_infinity).nonzero().squeeze(1)
        log_probs = log_probs[ok_idx]
        # choose the most promising results
        top_idx = log_probs.topk(k=min(log_probs.size(0), k), dim=0, sorted=False)[1]
        log_probs = log_probs[top_idx]
        # calculate the indices to choose from the original beam
        next_idx = ok_idx[top_idx]

        starts = starts[next_idx]
        cur_city = cur_city[next_idx]
        cur_group = cur_group[next_idx]
        used = used[next_idx]
        results = results[next_idx]
        dists = dists[next_idx]

        # normalize the probabilities
        log_probs = log_probs - log_probs.max()

        return starts, cur_city, cur_group, used, results, dists, log_probs

    def loop(self, starts, cur_city, cur_group, used, results, dists, log_probs, k, last=False):
        # starts:       Long(beam x (groups + 1))
        # cur_city:     Long(beam)
        # cur_group:    Long(beam)
        # used:         Byte(beam x cities)
        # results:      Long(beam x c x 2)
        # dists:        Float(beam x groups)
        # log_probs:    Float(beam)
        beam = log_probs.size(0)

        # calculate new starts
        # Long(beam x cities x (groups + 1))
        new_starts = starts.unsqueeze(1).expand(beam, self.cities, self.groups + 1)

        # calculate new cur_city
        new_cur_city = self.cities_range.unsqueeze(0).repeat(beam, 1)  # Long(beam x cities)

        # calculate new cur_group
        new_cur_group = cur_group.unsqueeze(1).repeat(1, self.cities)  # Long(beam x cities)

        # calculate new probs
        new_log_probs = log_probs.unsqueeze(1).repeat(1, self.cities)  # Float(beam x cities)
        # Float(beam x (groups * cities) x cities)
        cur_probs = self.log_probs.view(-1, self.cities).unsqueeze(0).expand(beam, -1, self.cities)
        probs_indices = cur_group * self.cities + cur_city  # Long(beam)
        cur_probs = cur_probs.gather(
            dim=1,
            index=probs_indices.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.cities)
        ).squeeze(1)  # Float(beam x cities)
        new_log_probs.add_(cur_probs)  # Float(beam x cities)

        # calculate new used
        new_used = used.unsqueeze(1).repeat(1, self.cities, 1)  # Byte(beam x cities x cities)

        # calculate new dists
        new_dists = dists.unsqueeze(1).expand(beam, self.cities, self.groups)  # Float(beam x cities x groups)
        dists_delta = self.dists[cur_city].unsqueeze(2).expand_as(new_dists)  # Float(beam x cities x groups)
        new_dists_mask = zeros(*new_dists.size()).scatter_(dim=2,
                                                           index=new_cur_group.unsqueeze(2),
                                                           src=ones(*new_dists.size()))  # Float(beam x cities x groups)
        new_dists = new_dists + dists_delta * new_dists_mask

        # zero probs if new_cur_city is already used
        used_mask = new_used.gather(dim=2, index=new_cur_city.unsqueeze(2)).squeeze(2)  # Byte(beam x cities)
        new_log_probs[used_mask == 1] = self.minus_infinity

        # fix new_cur_city and new_cur_group
        new_cur_group[:, 0] += 1  # Long(beam x cities)
        next_cities = new_cur_city.clone()  # Long(beam x cities)
        new_cur_city[:, 0] = starts.gather(dim=1, index=cur_group.unsqueeze(1) + 1).squeeze(1)  # Long(beam x cities)

        if not last:
            # zero probabilities of invalid samples, if their next_starts is to big
            new_log_probs[new_cur_group == self.groups] = self.minus_infinity

        # update new used
        # Byte(beam x cities x cities)
        new_used.scatter_(dim=2, index=next_cities.unsqueeze(2), src=ones(beam, self.cities, 1, type='byte'))
        new_used[:, :, 0] = 0  # Byte(beam x cities x cities)

        new_starts = new_starts.contiguous().view(-1, self.groups + 1)
        new_cur_city = new_cur_city.contiguous().view(-1)
        new_cur_group = new_cur_group.contiguous().view(-1)
        new_used = new_used.contiguous().view(-1, self.cities)
        new_dists = new_dists.contiguous().view(-1, self.groups)
        new_log_probs = new_log_probs.contiguous().view(-1)

        # leave only samples with positive probability
        ok_idx = (new_log_probs > self.minus_infinity).nonzero().squeeze(1)
        new_log_probs = new_log_probs[ok_idx]
        # choose the most promising results
        top_idx = new_log_probs.topk(k=min(new_log_probs.size(0), k), dim=0, sorted=False)[1]
        new_log_probs = new_log_probs[top_idx]
        # calculate the indices to choose from the original beam
        next_idx = ok_idx[top_idx]
        orig_idx = next_idx / self.cities

        new_starts = new_starts[next_idx]
        new_cur_city = new_cur_city[next_idx]
        new_cur_group = new_cur_group[next_idx]
        new_used = new_used[next_idx]
        new_dists = new_dists[next_idx]

        # calculate new_results
        next_cities = next_cities.contiguous().view(-1)[next_idx]
        new_match = torch.stack((cur_city[orig_idx], next_cities), dim=1)  # Long(beam x 2)
        new_results = torch.cat((results[orig_idx], new_match.unsqueeze(1)), dim=1)  # Long(beam x (c + 1) x 2)

        # normalize the probabilities
        new_log_probs = new_log_probs - new_log_probs.max()

        return new_starts, new_cur_city, new_cur_group, new_used, new_results, new_dists, new_log_probs
