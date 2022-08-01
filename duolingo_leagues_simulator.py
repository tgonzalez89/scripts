NUM_USERS = 1000001
SIM_LOOPS = 1000


class League:
    def __init__(self, max_per_bucket, promotion, demotion) -> None:
        self.max_per_bucket = max_per_bucket
        self.promotion = promotion
        self.demotion = demotion
        self.users = 0

    def num_full_buckets(self):
        return self.users // self.max_per_bucket

    def users_remainder(self):
        return self.users % self.max_per_bucket

    def calc_promote(self):
        total = self.promotion * self.num_full_buckets()
        users_remainder = self.users_remainder()
        if users_remainder < self.promotion:
            total += users_remainder
        else:
            total += self.promotion
        return total

    def calc_demote(self):
        total = self.demotion * self.num_full_buckets()
        users_remainder = self.users_remainder()
        not_demoted = self.max_per_bucket - self.demotion
        if users_remainder > not_demoted:
            total += users_remainder - not_demoted
        return total


leagues = (
    League(30, 15, 0),
    League(30, 15, 7),
    League(30, 10, 7),
    League(30, 10, 7),
    League(30, 10, 7),
    League(30, 10, 7),
    League(30, 10, 7),
    League(30, 10, 7),
    League(30, 5, 7),
    League(30, 0, 5),
)

leagues_tmp_results = [0]*10
result_str = ""

def promote():
    for i in range(0, 9):
        to_promote = leagues[i].calc_promote()
        leagues_tmp_results[i] -= to_promote
        leagues_tmp_results[i+1] += to_promote

def demote():
    for i in range(1, 10):
        to_demote = leagues[i].calc_demote()
        leagues_tmp_results[i] -= to_demote
        leagues_tmp_results[i-1] += to_demote

def apply():
    for i in range(10):
        leagues[i].users += leagues_tmp_results[i]
        leagues_tmp_results[i] = 0

def print_leagues(iter):
    global result_str
    print(f"{iter=}")
    result_str = ""
    for i in range(10):
        result_str += f"league={i}, users={leagues[i].users}\n"
    print(result_str)


if __name__ == "__main__":
    leagues[0].users = NUM_USERS
    prev_result_str = ""
    for i in range(SIM_LOOPS):
        prev_result_str = result_str
        print_leagues(i)
        if result_str == prev_result_str:
            break
        promote()
        demote()
        apply()
